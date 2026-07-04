# API Reference

The complete public API of Physiotrack. Every symbol on these pages is exported
from the top-level `physiotrack` package (or `physiotrack.signals`) and is part of
the supported, documented surface. Pages are generated directly from the source
docstrings, so they always match the installed version.

```python
import physiotrack as pt
```

## Predictors

Every image predictor exposes a single verb — `.predict(img)` — and is callable
(`model(img)`). Each returns a unified [`Result`](results.md) (or `list[Result]`
for a batch). Rendering is a method on the result, `result.plot()`.

| Class | Task | Returns |
| --- | --- | --- |
| [`Detection`](detection.md) | Person / face / object boxes | [`Result`](results.md#result) |
| [`Pose`](pose.md) | 2D whole-body / COCO keypoints | [`Result`](results.md#result) |
| [`Pose3D`](pose3d.md) | 2D → 3D pose lifting | tuple / 3D arrays |
| [`Segmentation`](segmentation.md) | Body-part / person / face masks | [`Result`](results.md#result) |
| [`Depth`](depth.md) | Monocular dense depth | [`DepthResult`](results.md#depthresult) |
| [`Face`](face.md) | Face detection + head orientation | [`Result`](results.md#result) |

## Building blocks

| Class / Module | Purpose |
| --- | --- |
| [Result objects](results.md) | `Result`, `Instance`, `Keypoints`, `Keypoint`, `DepthResult`, `TrackResult` |
| [`Video`](video.md) | End-to-end video orchestrator (detect → pose → track → overlay) |
| [`Tracker` / `TrackerConfig`](tracking.md) | Multi-object tracking (ByteTrack, StrongSORT, OCSort, BoostTrack) |
| [`Models`](models.md) | Model registry + weight auto-download |
| [Pose post-processing](pose-postprocessing.md) | Canonicalization + 3D evaluation metrics |
| [Signals](signals/index.md) | rPPG, motion features, joint angles / ROM, filters, plotting |
