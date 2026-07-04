# Plotting

Live and offline visualizers for signals derived from tracking — heart rate, rPPG
waveforms, joint angles, and keypoint motion. See the [Signals guide](../../guides/signals.md).

## RPPGPlotter

::: physiotrack.signals.RPPGPlotter

## HeartRatePlotter

::: physiotrack.signals.HeartRatePlotter

## HRVPlotter

::: physiotrack.signals.HRVPlotter

## RespirationPlotter

::: physiotrack.signals.RespirationPlotter

## JointAnglePlotter

::: physiotrack.signals.JointAnglePlotter

## KeypointMotionPlotter

::: physiotrack.signals.KeypointMotionPlotter

## RealTimePlotter

::: physiotrack.signals.RealTimePlotter

## EstimatorPanel (base)

Shared base for the estimator-backed vitals panels (rPPG / HR / HRV / respiration) —
wraps a [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator] and composites a
BGRA panel onto a frame.

::: physiotrack.signals.plotting._estimator_panel.EstimatorPanel
