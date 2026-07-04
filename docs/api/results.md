# Result Objects

The unified, typed object model returned by every predictor. A `Result` is an
iterable of [`Instance`](#instance) objects; each `Instance` carries the fields
relevant to its task (box, keypoints, mask, orientation). Depth returns a
[`DepthResult`](#depthresult); the tracker returns a [`TrackResult`](#trackresult).

Rendering is always a method on the result — `result.plot(...)` — so the same
inference output can be drawn many ways without re-running the model.

## Result

::: physiotrack.Result

## Instance

::: physiotrack.Instance

## Keypoints

::: physiotrack.Keypoints

## Keypoint

::: physiotrack.Keypoint

## DepthResult

::: physiotrack.DepthResult

## TrackResult

::: physiotrack.TrackResult
