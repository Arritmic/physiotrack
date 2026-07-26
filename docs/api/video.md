# Video Orchestrator

`Video` runs a full pipeline over a video source — detection, tracking, pose, segmentation,
face orientation, depth, and overlay rendering — with a single configurable object. See the
[Video Pipeline guide](../guides/video.md).

`run()` returns a [`VideoResults`][physiotrack.VideoResults] sequence of
[`FrameResult`][physiotrack.FrameResult] objects, so the per-frame subjects remain
[`Instance`][physiotrack.Instance] objects with named keypoints rather than plain dicts.

::: physiotrack.Video

## Capture helpers

Exported so a hand-written capture loop behaves the same way the orchestrator does — the
same orientation handling, and a writer that cannot silently fail when an encoder is
missing.

::: physiotrack.capture.resolve_rotation

::: physiotrack.capture.apply_rotation

::: physiotrack.capture.open_video_writer
