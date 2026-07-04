# Tracking

Multi-object tracking with a unified wrapper over ByteTrack, StrongSORT, OCSort, and
BoostTrack. Returns a [`TrackResult`](results.md#trackresult) whose instances carry a
persistent `id`. All backend hyperparameters are configured through
[`TrackerConfig`](#trackerconfig). See the [Tracking guide](../guides/tracking.md).

## Tracker

::: physiotrack.Tracker

## TrackerConfig

::: physiotrack.TrackerConfig
