"""Shared constants for the rPPG / pulse-analysis subsystem.

This module is the **single source of truth** for the analysis frequency bands and
the rPPG-method registry used across the ``physiotrack.signals.ppg`` package. Import
these here rather than re-declaring band edges or method maps in individual modules,
so the heart-rate, HRV, respiration and benchmarking code all agree on the same
definitions.

The heart-rate and HRV bands follow the standard cardiovascular conventions:

* HR search / band-pass band: 0.75-4.0 Hz (45-240 bpm), the range used throughout
  the rPPG literature (e.g. Wang et al., "Algorithmic Principles of Remote PPG",
  IEEE TBME 2017).
* HRV frequency bands (VLF/LF/HF): the boundaries recommended by the Task Force of
  the European Society of Cardiology and the North American Society of Pacing and
  Electrophysiology, "Heart rate variability: standards of measurement, physiological
  interpretation, and clinical use", *Circulation* 93(5), 1996.
* Respiration band: 0.10-0.50 Hz (6-30 breaths/min), a standard adult resting-to-
  active range (Charlton et al., "Breathing Rate Estimation From the ECG and PPG:
  A Review", IEEE Reviews in Biomedical Engineering, 2018).
"""

from .extraction import POS, CHROM, LGI, OMIT

__all__ = [
    "HR_BAND",
    "RESP_BAND",
    "HRV_VLF_BAND",
    "HRV_LF_BAND",
    "HRV_HF_BAND",
    "RPPG_METHODS",
    "DEFAULT_RPPG_METHOD",
]

#: Heart-rate search / band-pass band in Hz -- ``(lo, hi)`` == 45-240 bpm.
HR_BAND = (0.75, 4.0)

#: Respiration band in Hz -- ``(lo, hi)`` == 6-30 breaths/min.
RESP_BAND = (0.10, 0.50)

#: HRV very-low-frequency band in Hz (Task Force 1996).
HRV_VLF_BAND = (0.0033, 0.04)
#: HRV low-frequency band in Hz (Task Force 1996).
HRV_LF_BAND = (0.04, 0.15)
#: HRV high-frequency band in Hz (Task Force 1996).
HRV_HF_BAND = (0.15, 0.40)

#: The rPPG blood-volume-pulse extraction methods, keyed by upper-case name. This is
#: the one registry used by :class:`~physiotrack.signals.ppg.estimator.HeartRateEstimator`
#: and :func:`~physiotrack.signals.ppg.metrics.benchmark_rppg_methods`.
RPPG_METHODS = {"POS": POS, "CHROM": CHROM, "LGI": LGI, "OMIT": OMIT}

#: Recommended default rPPG method. POS (Wang et al. 2017) is the most robust of the
#: four to motion and illumination changes and is the physiotrack default.
DEFAULT_RPPG_METHOD = "POS"
