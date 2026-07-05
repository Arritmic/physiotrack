"""Heart-rate estimation and rPPG evaluation metrics.

Turns a blood-volume-pulse (BVP) signal into a heart rate (HR) and scores it
against a reference. The definitions follow the standard rPPG-evaluation
conventions used across the literature (Welch power-spectral-density peak for HR;
the de~Haan signal-to-noise ratio): the band is 0.65--4.0 Hz (39--240 bpm), HR is
the spectral peak in that band, and the SNR contrasts the power around the HR
fundamental and its first harmonic with the rest of the band. The implementation
here is original.
"""

import numpy as np
from scipy.signal import welch, detrend
from scipy.stats import pearsonr

from physiotrack.signals.ppg.constants import HR_BAND, RPPG_METHODS

__all__ = ["bvp_to_hr", "bvp_snr", "hr_errors", "benchmark_rppg_methods"]

# Default rPPG analysis band (Hz): 0.75--4.0 Hz == 45--240 bpm, sourced from the
# single band definition in ``constants.HR_BAND``. Override via the lo_hz/hi_hz args.
_LO_HZ, _HI_HZ = HR_BAND


def _welch_psd(x, fps, nfft=2048):
    """Detrended Welch PSD of a 1-D signal; returns (freqs_hz, power)."""
    x = detrend(np.asarray(x, dtype=float))
    nperseg = len(x)
    if nperseg < 2:
        return np.array([]), np.array([])
    freqs, power = welch(x, fs=fps, window="hann", nperseg=nperseg,
                         nfft=max(int(nfft), nperseg), detrend=False)
    return freqs, power


def bvp_to_hr(bvp, fps, win_sec=10.0, step_sec=1.0,
              lo_hz=_LO_HZ, hi_hz=_HI_HZ, nfft=2048):
    """Per-window heart rate (bpm) from a BVP signal.

    For each sliding window the HR is the frequency of the Welch-PSD peak inside
    ``[lo_hz, hi_hz]``, converted to beats per minute. If the signal is shorter
    than one window, a single window over the whole signal is used.

    Args:
        bvp (np.ndarray): 1-D blood-volume-pulse signal (e.g. from
            [`POS`][physiotrack.signals.POS] and a band-pass filter).
        fps (float): Sampling rate of ``bvp`` in Hz.
        win_sec (float, optional): Sliding-window length in seconds. Defaults to
            ``10.0``.
        step_sec (float, optional): Hop between window starts in seconds (min 1
            sample). Defaults to ``1.0``.
        lo_hz (float, optional): Lower band edge in Hz. Defaults to ``0.75``
            (45 bpm).
        hi_hz (float, optional): Upper band edge in Hz. Defaults to ``4.0``
            (240 bpm).
        nfft (int, optional): FFT length for the Welch PSD (raised to at least the
            window length). Defaults to ``2048``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(hr_bpm, times)`` where ``hr_bpm`` is the
            per-window heart rate in beats per minute and ``times`` are the
            window-centre timestamps in seconds. Both are empty arrays if the
            signal has fewer than 2 samples.
    """
    bvp = np.asarray(bvp, dtype=float)
    n = len(bvp)
    if n < 2:
        return np.array([]), np.array([])
    w = int(win_sec * fps)
    step = max(1, int(step_sec * fps))
    if n < w:
        starts, w = [0], n
    else:
        starts = range(0, n - w + 1, step)

    hrs, times = [], []
    for st in starts:
        freqs, power = _welch_psd(bvp[st:st + w], fps, nfft)
        band = (freqs >= lo_hz) & (freqs <= hi_hz)
        if not np.any(band):
            continue
        f_peak = freqs[band][int(np.argmax(power[band]))]
        hrs.append(f_peak * 60.0)
        times.append((st + w / 2.0) / fps)
    return np.array(hrs), np.array(times)


def bvp_snr(bvp, fps, ref_hr_bpm, lo_hz=0.5, hi_hz=_HI_HZ, half_bw_hz=0.1, nfft=2048):
    """Signal-to-noise ratio (dB) of a BVP given a reference HR (de Haan).

    Signal power is the PSD summed within ``+/- half_bw_hz`` of the reference HR
    fundamental and its first harmonic (2x HR); noise is the remaining power in
    ``[lo_hz, hi_hz]``. ``SNR = 10 log10(signal / noise)``.

    Args:
        bvp (np.ndarray): 1-D blood-volume-pulse signal.
        fps (float): Sampling rate of ``bvp`` in Hz.
        ref_hr_bpm (float): Reference heart rate in beats per minute defining the
            signal bands. If ``None`` or NaN the function returns ``np.nan``.
        lo_hz (float, optional): Lower band edge in Hz. Defaults to ``0.5``.
        hi_hz (float, optional): Upper band edge in Hz. Defaults to ``4.0``.
        half_bw_hz (float, optional): Half-width in Hz of the signal band around
            the fundamental and first harmonic. Defaults to ``0.1``.
        nfft (int, optional): FFT length for the Welch PSD. Defaults to ``2048``.

    Returns:
        float: Signal-to-noise ratio in decibels, or ``np.nan`` when the reference
            HR is missing or the band contains no usable power.
    """
    if ref_hr_bpm is None or np.isnan(ref_hr_bpm):
        return np.nan
    freqs, power = _welch_psd(bvp, fps, nfft)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    freqs, power = freqs[band], power[band]
    if power.size == 0:
        return np.nan
    f0 = ref_hr_bpm / 60.0
    sig_mask = (np.abs(freqs - f0) <= half_bw_hz) | (np.abs(freqs - 2.0 * f0) <= half_bw_hz)
    s_pow = float(np.sum(power[sig_mask]))
    n_pow = float(np.sum(power[~sig_mask]))
    if s_pow <= 0 or n_pow <= 0:
        return np.nan
    return 10.0 * np.log10(s_pow / n_pow)


def hr_errors(hr_est, hr_gt):
    """Agreement metrics between estimated and reference HR series (bpm).

    The two series are truncated to a common length and reduced to their finite,
    non-zero-reference samples before scoring.

    Args:
        hr_est (np.ndarray): Estimated heart-rate series in beats per minute.
        hr_gt (np.ndarray): Reference (ground-truth) heart-rate series in beats
            per minute.

    Returns:
        dict: ``{"MAE", "RMSE", "MAPE", "Pearson"}`` where ``MAE`` and ``RMSE`` are
            in bpm, ``MAPE`` is a percentage, and ``Pearson`` is the correlation
            coefficient. All values are ``np.nan`` if fewer than 2 valid samples
            remain.

    Example:
        ```python
        from physiotrack.signals import hr_errors
        scores = hr_errors(hr_est, hr_gt)
        print(scores["MAE"], scores["Pearson"])
        ```
    """
    e = np.asarray(hr_est, dtype=float)
    g = np.asarray(hr_gt, dtype=float)
    m = min(len(e), len(g))
    e, g = e[:m], g[:m]
    keep = ~(np.isnan(e) | np.isnan(g)) & (g != 0)
    e, g = e[keep], g[keep]
    if e.size < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "Pearson": np.nan}
    diff = e - g
    out = {
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "MAPE": float(np.mean(np.abs(diff / g)) * 100.0),
    }
    try:
        out["Pearson"] = float(pearsonr(e, g)[0])
    except Exception:
        out["Pearson"] = np.nan
    return out


def benchmark_rppg_methods(rgb_trace, fps, ref_hr_bpm=None, hr_band=HR_BAND):
    """Compare all rPPG extraction methods on one RGB skin trace.

    Runs each method in
    [`RPPG_METHODS`][physiotrack.signals.ppg.constants.RPPG_METHODS] (POS, CHROM, LGI,
    OMIT) over the same RGB trace, band-passes each blood-volume pulse to ``hr_band``,
    and scores the resulting heart rate and de~Haan SNR. Use this to justify the
    default method choice on your own data. POS is the physiotrack default as the most
    motion- and illumination-robust of the four (Wang et al., 2017).

    Args:
        rgb_trace (np.ndarray): RGB skin colour trace of shape ``(3, N)`` (rows R, G, B).
        fps (float): Sampling rate of the trace in Hz.
        ref_hr_bpm (float, optional): Reference heart rate in bpm; when given, an
            absolute error ``AE`` (bpm) is added per method. Defaults to ``None``.
        hr_band (tuple[float, float], optional): HR band-pass / search band in Hz.
            Defaults to [`HR_BAND`][physiotrack.signals.ppg.constants.HR_BAND].

    Returns:
        dict: ``{method_name: {"hr": bpm, "snr": dB, "AE": bpm_or_nan}}`` for each of
            POS / CHROM / LGI / OMIT.

    Example:
        ```python
        from physiotrack.signals import benchmark_rppg_methods
        scores = benchmark_rppg_methods(rgb_trace, fps=30.0, ref_hr_bpm=72.0)
        best = min(scores, key=lambda m: scores[m]["AE"])
        print("best method:", best)
        ```

    See Also:
        [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator]: the streaming
            estimator that wraps a single chosen method.
    """
    from physiotrack.signals.filters import bandpass_filter
    lo, hi = float(hr_band[0]), float(hr_band[1])
    trace = np.asarray(rgb_trace, dtype=float)
    out = {}
    for name, cls in RPPG_METHODS.items():
        bvp = np.asarray(cls(fps).apply(trace), dtype=float).ravel()
        try:
            bvp = bandpass_filter(bvp, lo, hi, fps)
        except Exception:
            pass
        hr, _ = bvp_to_hr(bvp, fps, lo_hz=lo, hi_hz=hi)
        hr_val = float(hr[-1]) if hr.size else np.nan
        # SNR is measured about the *reference* HR when one is supplied (de Haan &
        # Jeanne, 2013), so it fairly ranks methods; only fall back to the method's
        # own peak when no ground truth is available.
        snr_ref = ref_hr_bpm if (ref_hr_bpm is not None and np.isfinite(ref_hr_bpm)) else hr_val
        snr = bvp_snr(bvp, fps, snr_ref, hi_hz=hi)
        ae = abs(hr_val - ref_hr_bpm) if (ref_hr_bpm is not None and np.isfinite(hr_val)) else np.nan
        out[name] = {"hr": hr_val, "snr": snr, "AE": ae}
    return out
