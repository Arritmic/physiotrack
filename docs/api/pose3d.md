# Pose 3D

Lifts 2D keypoint sequences to 3D (MotionBERT, DDHPose) with optional canonical-view
normalization. See [3D Pose & Canonicalization](../guides/pose3d.md) and the related
[post-processing utilities](pose-postprocessing.md).

!!! note
    `Pose3D` is imported lazily (`import physiotrack as pt; pt.Pose3D`) so that
    `import physiotrack` does not pull in heavy 3D-rendering dependencies.

::: physiotrack.pose.pose3D.Pose3D
