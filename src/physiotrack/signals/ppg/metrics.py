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

__all__ = ["bvp_to_hr", "bvp_snr", "hr_errors"]

# Default rPPG analysis band (Hz): 0.75--4.0 Hz == 45--240 bpm. Matches the
# physiotrack band-pass default; override via the lo_hz/hi_hz arguments.
_LO_HZ, _HI_HZ = 0.75, 4.0


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
    ``[lo_hz, hi_hz]``, converted to beats per minute. Returns ``(hr_bpm, times)``
    where ``times`` are the window-centre timestamps in seconds. If the signal is
    shorter than one window, a single window over the whole signal is used.
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

    Returns ``{MAE, RMSE, MAPE, Pearson}`` over the temporally aligned, finite
    samples (MAE/RMSE in bpm, MAPE in %).
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
