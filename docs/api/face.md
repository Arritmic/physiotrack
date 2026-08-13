# Face & Head Orientation

Face detection presets and 6-DoF head-orientation (yaw/pitch/roll) estimation.
See the [Face guide](../guides/face.md).

!!! note "`Face()` and `Detection.Face()`"
    Top-level [`Face`][physiotrack.Face] is the facial-pipeline entry point and
    returns `Result(task="face")`. [`Detection.Face`][physiotrack.Detection.Face]
    uses the same default face checkpoint through the generic detection namespace
    and returns `Result(task="detect")`. See the
    [runnable face examples](../guides/face-examples.md#api-choice-face-or-detectionface)
    for the practical distinction.

## Face

::: physiotrack.Face

## VRFace

::: physiotrack.VRFace

## FaceOrientation

::: physiotrack.FaceOrientation
