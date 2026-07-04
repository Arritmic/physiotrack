# Signals

The `physiotrack.signals` subsystem turns tracked pixels into physiological and
biomechanical signals, and provides the DSP, metrics, and plotting utilities around
them.

```python
from physiotrack.signals import POS, HeartRateEstimator, compute_rom_angles, RPPGPlotter
```

| Module | What it provides |
| --- | --- |
| [rPPG / Heart Rate](rppg.md) | Remote-PPG extraction (`POS`, `CHROM`, `LGI`, `OMIT`), `HeartRateEstimator`, skin ROI (`FaceSkinExtractor`), HR metrics |
| [Motion & Features](motion.md) | Keypoint sequences, centroids, joint angles, range-of-motion (ROM) |
| [Filters](filters.md) | Band-pass / notch / high- & low-pass, detrending, SNR |
| [Normalization](normalize.md) | Min-max, z-score, robust, quantile, power-transform, … |
| [Signal Metrics](evaluate.md) | PLV, phase synchrony, RMSE, cross-correlation, DTW |
| [Plotting](plotting.md) | Real-time and offline plotters for HR, rPPG, angles, keypoints |

See the [Signals guide](../../guides/signals.md) for end-to-end examples.
