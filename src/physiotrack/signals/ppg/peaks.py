"""Systolic-peak detection and RR-interval (inter-beat-interval) extraction.

Turns a blood-volume-pulse (BVP) waveform -- e.g. the band-passed output of
[`POS`][physiotrack.signals.POS] via
[`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator] -- into the series
of systolic peak locations and the RR-interval (a.k.a. inter-beat-interval, IBI)
series that heart-rate-variability analysis is built on.

Peaks are found with :func:`scipy.signal.find_peaks`, constrained so successive
beats are at least ``1 / hi_hz`` seconds apart (i.e. no faster than the top of the
heart-rate band). RR intervals are the successive peak-to-peak times, reported in
milliseconds following the HRV convention (Task Force of the ESC/NASPE, "Heart rate
variability: standards of measurement, physiological interpretation and clinical
use", *Circulation* 93(5), 1996).
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from physiotrack.signals.ppg.constants import HR_BAND

__all__ = ["detect_pulse_peaks", "bvp_to_rri"]


def detect_pulse_peaks(bvp: np.ndarray, fps: float,
                       hr_band: Tuple[float, float] = HR_BAND,
                       prominence: Optional[float] = None) -> np.ndarray:
    """Locate systolic peaks in a blood-volume-pulse waveform.

    Runs :func:`scipy.signal.find_peaks` with a minimum inter-peak distance derived
    from the upper edge of ``hr_band`` (``distance = fps / hi_hz`` samples), so two
    detected beats are never closer than the fastest plausible heart rate. Supply the
    BVP already band-passed to the heart-rate band for best results.

    Args:
        bvp (np.ndarray): 1-D blood-volume-pulse signal (systolic upstrokes positive).
        fps (float): Sampling rate of ``bvp`` in Hz.
        hr_band (tuple[float, float], optional): ``(lo_hz, hi_hz)`` heart-rate band;
            only ``hi_hz`` is used, to set the minimum beat spacing. Defaults to
            [`HR_BAND`][physiotrack.signals.ppg.constants.HR_BAND] ``(0.75, 4.0)``.
        prominence (float, optional): Minimum peak prominence passed to
            ``find_peaks``. ``None`` (default) applies no prominence constraint.

    Returns:
        np.ndarray: Integer sample indices of the detected systolic peaks (empty if
            fewer than one peak is found).

    Example:
        ```python
        from physiotrack.signals import detect_pulse_peaks
        peaks = detect_pulse_peaks(bvp, fps=30.0)   # bvp: band-passed 1-D signal
        ```

    See Also:
        [`bvp_to_rri`][physiotrack.signals.bvp_to_rri]: builds the RR-interval series
            from these peaks.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a positive, finite number, got {fps!r}.")
    bvp = np.asarray(bvp, dtype=float).ravel()
    if bvp.size < 2:
        return np.array([], dtype=int)
    if not np.all(np.isfinite(bvp)):
        raise ValueError("bvp contains non-finite values (NaN/inf); supply a finite, "
                         "gap-filled signal (find_peaks silently skips beats near NaNs).")
    hi_hz = float(hr_band[1])
    # Floor (not round) so beats up to the top of the band are admitted: at the band
    # edge round() would over-space and merge legitimate fast beats.
    distance = max(1, int(fps / hi_hz))
    peaks, _ = find_peaks(bvp, distance=distance, prominence=prominence)
    return peaks.astype(int)


def bvp_to_rri(bvp: np.ndarray, fps: float,
               hr_band: Tuple[float, float] = HR_BAND,
               prominence: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the RR-interval (inter-beat-interval) series from a BVP waveform.

    Detects systolic peaks with
    [`detect_pulse_peaks`][physiotrack.signals.detect_pulse_peaks] and returns the
    successive peak-to-peak durations in milliseconds, together with the time of the
    later beat of each interval (seconds). This ``(rri_ms, t_sec)`` pair is the input
    expected by the HRV functions in
    [`physiotrack.signals.ppg.hrv`][physiotrack.signals.compute_hrv] and by the
    respiratory-sinus-arrhythmia estimator
    [`respiration_from_rri`][physiotrack.signals.respiration_from_rri].

    Args:
        bvp (np.ndarray): 1-D blood-volume-pulse signal (band-passed to the HR band).
        fps (float): Sampling rate of ``bvp`` in Hz.
        hr_band (tuple[float, float], optional): ``(lo_hz, hi_hz)`` heart-rate band
            forwarded to the peak detector. Defaults to
            [`HR_BAND`][physiotrack.signals.ppg.constants.HR_BAND].
        prominence (float, optional): Minimum peak prominence forwarded to the peak
            detector. Defaults to ``None``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(rri_ms, t_sec)`` where ``rri_ms`` are the
            inter-beat intervals in milliseconds and ``t_sec`` are the corresponding
            beat timestamps in seconds (the time of the second peak of each interval).
            Both are empty when fewer than two peaks are detected.

    Example:
        ```python
        from physiotrack.signals import bvp_to_rri, compute_hrv
        rri_ms, t_sec = bvp_to_rri(bvp, fps=30.0)
        hrv = compute_hrv(rri_ms)
        print(hrv["RMSSD"], hrv["SDNN"])
        ```

    See Also:
        [`correct_rr_artifacts`][physiotrack.signals.correct_rr_artifacts]: clean the
            returned ``rri_ms`` before computing HRV.
    """
    peaks = detect_pulse_peaks(bvp, fps, hr_band, prominence)
    if peaks.size < 2:
        return np.array([]), np.array([])
    peak_t = peaks / float(fps)
    rri_ms = np.diff(peak_t) * 1000.0
    t_sec = peak_t[1:]
    return rri_ms, t_sec
