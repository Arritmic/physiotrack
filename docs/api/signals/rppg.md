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
