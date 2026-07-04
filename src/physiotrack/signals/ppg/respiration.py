"""Respiration-rate estimation from a pulse (rPPG) signal.

Breathing modulates the cardiac pulse three ways (Charlton et al., "Breathing Rate
Estimation From the ECG and PPG: A Review", *IEEE Reviews in Biomedical Engineering*
11:2-20, 2018):

* **RIAV** -- respiratory-induced *amplitude* variation: the pulse amplitude waxes and
  wanes with breathing. Recovered here from the Hilbert amplitude envelope, and robust
  to the heart-rate band-pass filtering the rPPG pipeline already applies.
* **RIIV** -- respiratory-induced *intensity* variation: slow baseline wander of the
  raw skin-intensity trace (requires the *un-filtered* signal).
* **RIFV** -- respiratory-induced *frequency* variation: respiratory sinus arrhythmia
  (RSA), the beat-to-beat heart-rate oscillation, recovered from the RR-interval series
  (see [`respiration_from_rri`][physiotrack.signals.respiration_from_rri]).

Each route reduces to a respiratory modulation waveform whose dominant frequency inside
the respiration band ([`RESP_BAND`][physiotrack.signals.ppg.constants.RESP_BAND],
0.10-0.50 Hz == 6-30 breaths/min) is the breathing rate. Because there is no single
canonical reference implementation, these functions are validated against synthetic
signals with a known modulation frequency in the test suite.
"""

import numpy as np
from scipy.signal import welch, hilbert
from scipy.interpolate import interp1d

from physiotrack.signals.ppg.constants import RESP_BAND
from physiotrack.signals.filters import bandpass_filter

__all__ = [
    "respiration_rate_from_signal",
    "respiration_from_pulse",
    "respiration_from_rri",
]


def respiration_rate_from_signal(mod, fs, resp_band=RESP_BAND, nfft=4096):
    """Respiration rate (breaths/min) from a respiratory modulation waveform.

    The single source of truth for turning any respiratory modulation signal (a pulse
    amplitude envelope, a baseline-wander trace, an interpolated tachogram, or a
    chest-motion signal) into a rate: the waveform is mean-removed, band-passed to
    ``resp_band``, and the rate is the frequency of the Welch-PSD peak inside the band.

    Args:
        mod (np.ndarray): 1-D respiratory modulation waveform.
        fs (float): Sampling rate of ``mod`` in Hz.
        resp_band (tuple[float, float], optional): ``(lo_hz, hi_hz)`` respiration band.
            Defaults to [`RESP_BAND`][physiotrack.signals.ppg.constants.RESP_BAND].
        nfft (int, optional): FFT length for the Welch PSD (raised to at least the
            signal length) for fine low-frequency resolution. Defaults to ``4096``.

    Returns:
        tuple[float, np.ndarray]: ``(rate_bpm, filtered)`` -- the respiration rate in
            breaths per minute and the band-passed modulation waveform. ``rate_bpm`` is
            ``np.nan`` when the signal is too short or has no in-band power.

    Example:
        ```python
        from physiotrack.signals import respiration_rate_from_signal
        rate, wave = respiration_rate_from_signal(chest_y, fs=30.0)
        ```
    """
    mod = np.asarray(mod, dtype=float).ravel()
    mod = mod[np.isfinite(mod)]
    if mod.size < 8:
        return np.nan, np.asarray(mod)
    mod = mod - np.mean(mod)
    lo, hi = float(resp_band[0]), float(resp_band[1])
    try:
        filtered = bandpass_filter(mod, lo, hi, fs, order=4)
    except Exception:
        filtered = mod
    nperseg = len(filtered)
    freqs, psd = welch(filtered, fs=fs, window="hann", nperseg=nperseg,
                       nfft=max(int(nfft), nperseg), detrend="constant")
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band):
        return np.nan, filtered
    f_peak = freqs[band][int(np.argmax(psd[band]))]
    return float(f_peak * 60.0), filtered


def respiration_from_pulse(bvp, fps, method="riav", resp_band=RESP_BAND):
    """Estimate respiration rate from a pulse (BVP) signal.

    Args:
        bvp (np.ndarray): 1-D pulse signal. For ``method="riav"`` this may be the
            heart-rate-band-passed blood-volume pulse (the amplitude envelope survives
            band-pass filtering); for ``method="riiv"`` pass the *raw* skin-intensity
            trace, since baseline wander is removed by band-passing.
        fps (float): Sampling rate of ``bvp`` in Hz.
        method (str, optional): Modulation to use -- ``"riav"`` (amplitude, via Hilbert
            envelope) or ``"riiv"`` (baseline intensity). Defaults to ``"riav"``.
        resp_band (tuple[float, float], optional): Respiration band in Hz. Defaults to
            [`RESP_BAND`][physiotrack.signals.ppg.constants.RESP_BAND].

    Returns:
        tuple[np.ndarray, float]: ``(resp_wave, rate_bpm)`` -- the band-passed
            respiratory modulation waveform and the respiration rate in breaths/min
            (``np.nan`` if unavailable).

    Raises:
        ValueError: If ``method`` is not ``"riav"`` or ``"riiv"``.

    Example:
        ```python
        from physiotrack.signals import respiration_from_pulse
        resp_wave, rr = respiration_from_pulse(bvp, fps=30.0, method="riav")
        print(f"{rr:.1f} breaths/min")
        ```

    See Also:
        [`respiration_from_rri`][physiotrack.signals.respiration_from_rri]: the
            RSA-based (frequency-modulation) route.
    """
    bvp = np.asarray(bvp, dtype=float).ravel()
    if bvp.size < 8:
        return np.array([]), np.nan
    method = method.lower()
    if method == "riav":
        # Amplitude envelope of the pulse; its slow variation is the RIAV signal.
        mod = np.abs(hilbert(bvp - np.mean(bvp)))
    elif method == "riiv":
        # Baseline intensity itself (expects the raw, un-band-passed trace).
        mod = bvp
    else:
        raise ValueError(f"Unknown method {method!r}; choose 'riav' or 'riiv'.")
    rate, wave = respiration_rate_from_signal(mod, fps, resp_band)
    return wave, rate


def respiration_from_rri(rri_ms, t_sec=None, resp_band=RESP_BAND, interpolation_rate=4.0):
    """Estimate respiration rate from RR-interval variation (RSA / RIFV route).

    Resamples the RR-interval series onto an evenly-spaced tachogram and reads the
    respiration rate from its dominant in-band frequency -- respiratory sinus
    arrhythmia makes the heart rate oscillate at the breathing frequency.

    Args:
        rri_ms (np.ndarray): RR-interval series in milliseconds (e.g. from
            [`bvp_to_rri`][physiotrack.signals.bvp_to_rri]).
        t_sec (np.ndarray, optional): Beat timestamps in seconds, one per interval. If
            ``None``, cumulative RR times are used. Defaults to ``None``.
        resp_band (tuple[float, float], optional): Respiration band in Hz. Defaults to
            [`RESP_BAND`][physiotrack.signals.ppg.constants.RESP_BAND].
        interpolation_rate (float, optional): Tachogram resampling rate in Hz. Defaults
            to ``4.0``.

    Returns:
        tuple[np.ndarray, float]: ``(resp_wave, rate_bpm)`` -- the band-passed
            tachogram-derived waveform and the respiration rate in breaths/min
            (``np.nan`` if unavailable).

    Example:
        ```python
        from physiotrack.signals import bvp_to_rri, respiration_from_rri
        rri_ms, t_sec = bvp_to_rri(bvp, fps=30.0)
        _, rr = respiration_from_rri(rri_ms, t_sec)
        ```
    """
    rri = np.asarray(rri_ms, dtype=float)
    if rri.size < 8:
        return np.array([]), np.nan
    if t_sec is None:
        t = np.cumsum(rri) / 1000.0
    else:
        t = np.asarray(t_sec, dtype=float)
    m = np.isfinite(rri) & np.isfinite(t)
    rri, t = rri[m], t[m]
    if rri.size < 8:
        return np.array([]), np.nan
    fs = float(interpolation_rate)
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    if t_uniform.size < 8:
        return np.array([]), np.nan
    tachogram = interp1d(t, rri, kind="cubic", fill_value="extrapolate")(t_uniform)
    rate, wave = respiration_rate_from_signal(tachogram, fs, resp_band)
    return wave, rate
