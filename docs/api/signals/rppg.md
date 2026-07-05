# rPPG / Heart Rate

Remote photoplethysmography: recover a blood-volume-pulse (BVP) signal and heart rate
from an RGB skin trace. The four extraction methods take an RGB trace of shape
`(3, N)` (rows R, G, B). `HeartRateEstimator` is the high-level, streaming entry point;
`FaceSkinExtractor` produces the skin ROI. See the [Signals guide](../../guides/signals.md).

## HeartRateEstimator

::: physiotrack.signals.HeartRateEstimator

## Extraction methods

::: physiotrack.signals.POS

::: physiotrack.signals.CHROM

::: physiotrack.signals.LGI

::: physiotrack.signals.OMIT

## Skin extraction

::: physiotrack.signals.FaceSkinExtractor

::: physiotrack.signals.FaceParsing

## Heart-rate metrics

::: physiotrack.signals.bvp_to_hr

::: physiotrack.signals.bvp_snr

::: physiotrack.signals.hr_errors

::: physiotrack.signals.benchmark_rppg_methods

## RR intervals & artefact correction

Turn the blood-volume pulse into an RR-interval (inter-beat) series and clean it with
the Lipponen &amp; Tarvainen (2019) beat classifier before HRV analysis.

::: physiotrack.signals.detect_pulse_peaks

::: physiotrack.signals.bvp_to_rri

::: physiotrack.signals.find_rr_artifacts

::: physiotrack.signals.correct_rr_artifacts

## Heart-rate variability (HRV)

Time-, frequency- and non-linear-domain HRV indices (Task Force 1996; Brennan 2001;
Richman &amp; Moorman 2000), validated against closed-form references (and NeuroKit2 when
it is installed) in the test suite.

::: physiotrack.signals.compute_hrv

::: physiotrack.signals.hrv_time

::: physiotrack.signals.hrv_frequency

::: physiotrack.signals.hrv_nonlinear

::: physiotrack.signals.sample_entropy

::: physiotrack.signals.approximate_entropy

::: physiotrack.signals.hrv_errors

## Respiration

Contactless respiration rate from the pulse (RIAV / RSA) or from chest/shoulder motion.

::: physiotrack.signals.respiration_from_pulse

::: physiotrack.signals.respiration_from_rri

::: physiotrack.signals.respiration_rate_from_signal

::: physiotrack.signals.respiration_from_motion

## Constants & default bands

Default frequency bands (Hz) and the rPPG method registry shared across the HR / HRV /
respiration functions.

::: physiotrack.signals.ppg.constants
    options:
      members:
        - HR_BAND
        - RESP_BAND
        - HRV_VLF_BAND
        - HRV_LF_BAND
        - HRV_HF_BAND
        - RPPG_METHODS
        - DEFAULT_RPPG_METHOD
      show_if_no_docstring: true
      show_source: false
