# 3D Pose & Canonicalization

Lift a video's 2D keypoints into 3D joint positions, then optionally reorient
them to a fixed **canonical view** so downstream analysis is invariant to how the
subject was facing the camera. This is the bridge from
[2D pose](pose.md) to viewpoint-independent 3D motion.

The pipeline has three stages, each usable on its own:

1. **Lift 2D → 3D** with [`Pose3D`][physiotrack.pose.pose3D.Pose3D] (MotionBERT or DDHPose).
2. **Canonicalize** the 3D poses with [`canonicalize_pose`][physiotrack.canonicalize_pose] / [`PoseCanonicalizer`][physiotrack.PoseCanonicalizer] (GEOMETRIC or 3DPCNet).
3. **Evaluate** against ground truth with MPJPE / PA-MPJPE / rotation-error metrics.

All 3D poses use the **Human3.6M 17-joint** layout with the pelvis at index `0`,
and arrays have shape `(N, 17, 3)` — `N` frames, 17 joints, `(x, y, z)`.

## Quick start

`Pose3D.predict` takes 2D keypoints **in memory** and returns a
[`Pose3DResult`][physiotrack.Pose3DResult], like every other predictor. Feed it the
`VideoResults` from a 2D pass and the keypoints never touch the filesystem:

```python
import physiotrack as pt

View = pt.CanonicalView

# 1. 2D pass
results = pt.Video(source="clip.mp4", pose=pt.Pose.Person()).run()

# 2. lift to 3D
pose3d = pt.Pose3D(
    model=pt.Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite,
    device="cuda",
)
poses = pose3d.predict(
    results,
    fps=30,
    canonical_view=View.FRONT,     # optional: canonicalize during lifting
)

poses.poses.shape                  # (N, 17, 3)
poses.by_name("left_wrist")        # (N, 3) trajectory of one joint
poses[0].shape                     # (17, 3) one frame
```

`predict` also accepts a plain `(N, 17, 2)` / `(N, 17, 3)` COCO-17 array, so a
sequence from any source can be lifted.

!!! info "Lifting is sequence-level and single-subject"

    A temporal lifter needs a window of `clip_len` 2D frames to produce each 3D frame,
    so the whole sequence is passed at once — a single frame cannot be lifted in
    isolation. When the input carries several subjects per frame, the first is used.
    Frames with no detection contribute a zero-confidence pose so the output stays
    aligned with the source video rather than silently shortening.

!!! warning "Coordinates are relative, not metric"

    Output is root-relative and unitless unless the estimator was built with
    `pixel=True`. It is not comparable in scale between videos. `Pose3DResult.to_dict()`
    records this so a downstream consumer cannot mistake it for millimetres.

### Working from a JSON file instead

When the 2D pass and the lifting pass are separate steps, use `predict_json`. It reads
the 2D-pose JSON, lifts it, and returns the per-frame records augmented with 3D
keypoints alongside the result — and, with `out_path`, writes the rendered `.mp4`, the
`.npy` array, and a `*_with_3d_keypoints.json`:

```python
pose3d = pt.Pose3D(
    model=pt.Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite,
    device="cuda",
    render_video=True,     # write a 3D .mp4 when out_path is set
    save_npy=True,         # write the raw (N,17,3) array as .npy
)
frames_data, poses = pose3d.predict_json(
    "output/clip_result.json", "clip.mp4",
    out_path="output/", canonical_view=View.FRONT,
)
```

## Available models

### Lifting backends (`pt.Pose3D`)

The backend is inferred from the model enum.

| Model enum | Backend | Description |
| --- | --- | --- |
| `Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite` | MotionBERT | Transformer lifter (lite, global). Default when `model=None`. |
| `Models.Pose3D.MotionBERT.mb_ft_h36m` | MotionBERT | Full fine-tuned H36M variant. |
| `Models.Pose3D.MotionBERT.mb_train_h36m` | MotionBERT | Trained-from-scratch H36M variant. |
| `Models.Pose3D.DDH.best` | DDHPose (`DDH`) | Diffusion-based lifter with sampling controls. |

=== "MotionBERT"

    ```python
    import physiotrack as pt

    pose3d = pt.Pose3D(
        model=pt.Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite,
        device="cuda", clip_len=243,
    )
    poses = pose3d.predict(results, fps=30)
    ```

=== "DDHPose"

    ```python
    import physiotrack as pt

    pose3d = pt.Pose3D(
        model=pt.Models.Pose3D.DDH.best,
        device="cuda",
        num_proposals=10, sampling_timesteps=5,
    )
    # DDHPose normalises pixel coordinates, so it needs the source frame size.
    poses = pose3d.predict(results, fps=30, frame_size=(1920, 1080), batch_size=8)
    ```

See the [Model Zoo](../model-zoo.md) for weights and download details.

### Canonicalization methods (`Models.Pose3D.Canonicalizer.Models`)

| Method | Enum | Views | Notes |
| --- | --- | --- | --- |
| **Geometric** | `GEOMETRIC` (or `None`) | FRONT / BACK / LEFT_SIDE / RIGHT_SIDE | Closed-form, deterministic, no weights. |
| **3DPCNet** | `_3DPCNetS2`, `_3DPCNetS3`, `_3DPCNetTC48_byCam`, `_3DPCNetTC48_byAction` | FRONT only | Learned network; weights auto-download on first use. |

## Canonicalization

Canonicalization reorients each 3D pose so the subject faces a fixed direction,
removing global rotation. The four canonical views are the members of
`pt.CanonicalView`: `FRONT`, `BACK`, `LEFT_SIDE`,
`RIGHT_SIDE`.

- **GEOMETRIC** fits a torso plane from the shoulder + hip joints, aligns its
  normal with the camera axis and the shoulders with the X-axis (front view),
  then rotates about Y for the back/side views. Deterministic and dependency-free.
- **3DPCNet** is a learned network that regresses the canonicalizing rotation.
  It supports the front view only — other views warn and fall back to front.

The recommended entry point is the module-level
[`canonicalize_pose`][physiotrack.canonicalize_pose], which dispatches to the
right method based on `model`:

```python
import numpy as np
import physiotrack as pt

Canon = pt.Models.Pose3D.Canonicalizer
poses = np.random.randn(100, 17, 3)      # (N, 17, 3), H36M order

# Geometric — any of the four views
front = pt.canonicalize_pose(poses, model=Canon.Models.GEOMETRIC, view=Canon.View.FRONT)
left  = pt.canonicalize_pose(poses, view="left_side")   # GEOMETRIC + string view

# Learned 3DPCNet (auto-downloads weights; front only)
dpcnet = pt.canonicalize_pose(poses, model=Canon.Models._3DPCNetS2, view="front")

# Also get the rotation matrices (N, 3, 3)
canonical, rotation = pt.canonicalize_pose(
    poses, model=Canon.Models.GEOMETRIC, view="front", return_rotation=True,
)
```

You can canonicalize three ways:

=== "During lifting"

    ```python
    poses = pose3d.predict(
        results, fps=30,
        canonical_view=Canon.View.FRONT,
        canonical_model=Canon.Models.GEOMETRIC,
    )
    ```

=== "On an in-memory array"

    ```python
    canonical = pt.canonicalize_pose(
        poses.poses, model=Canon.Models.GEOMETRIC, view=Canon.View.FRONT,
    )
    ```

=== "From a file"

    ```python
    # From a saved .npy array
    canonical = pt.PoseCanonicalizer.process_npy_file(
        "output/X3D.npy", output_path="output/X3D_canonical.npy",
        view=Canon.View.FRONT, model=Canon.Models.GEOMETRIC,
    )

    # Or from a detection-results JSON with keypoints_3d
    data = pt.PoseCanonicalizer.process_json_file(
        "output/..._with_3d_keypoints.json", output_path="output/canonical.json",
        view="front", model=Canon.Models.GEOMETRIC,
    )
    ```

!!! note "3DPCNet coordinate format"
    3DPCNet works in an axis-remapped, pelvis-centered coordinate frame. By
    default `apply_transform=True` converts standard-format input for you; pass
    `apply_transform=False` only when your data is already in 3DPCNet format.

## Evaluation metrics

The [pose post-processing API](../api/pose-postprocessing.md) provides standard
3D-pose and canonicalization metrics (all accept NumPy arrays or torch tensors):

| Function | Measures |
| --- | --- |
| [`calculate_mpjpe`][physiotrack.calculate_mpjpe] | Mean Per Joint Position Error — unaligned Euclidean joint error. |
| [`calculate_pampjpe`][physiotrack.calculate_pampjpe] | Procrustes-aligned MPJPE — rigid + scale aligned, isolates pose *shape*. |
| [`calculate_rotation_error`][physiotrack.calculate_rotation_error] | Mean rotation error (`"frobenius"` default, or `"geodesic"` for a true angle). |
| [`evaluate_pose_predictions`][physiotrack.evaluate_pose_predictions] | Bundles MPJPE + PA-MPJPE + per-joint stats, scaled (e.g. ×1000 → mm). |
| [`evaluate_canonicalization`][physiotrack.evaluate_canonicalization] | The above plus a direct L2 pose error and (optional) rotation error. |

```python
import physiotrack as pt

preds = ...   # (N, 17, 3) predicted 3D poses
gts   = ...   # (N, 17, 3) ground-truth 3D poses

metrics = pt.evaluate_pose_predictions(preds, gts, scale=1000.0)  # meters -> mm
print(f"MPJPE {metrics['mpjpe']:.2f} mm, PA-MPJPE {metrics['pampjpe']:.2f} mm")

# Canonicalization quality, incl. rotation error
canonical, rotation = pt.canonicalize_pose(
    preds, model=pt.Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
    view="front", return_rotation=True,
)
scores = pt.evaluate_canonicalization(
    canonical, gt_canonical,
    pred_rotation=rotation, gt_rotation=gt_rotation, scale=1000.0,
)
print(scores["mpjpe"], scores.get("rotation_error_deg"))
```

!!! tip "MPJPE vs PA-MPJPE"
    MPJPE applies **no** alignment, so predictions and ground truth must share
    the same frame, scale, and root. PA-MPJPE first solves a similarity transform
    (rotation + translation + scale) per sample, so it scores the pose shape
    independent of orientation and size.

## Recipes & tips

!!! example "End-to-end: 2D → 3D"
    Run a [`Pose`][physiotrack.Pose] predictor over your video (writing a
    `*_result.json`), then feed that JSON and the video into `Pose3D.predict`.
    See the [2D Pose guide](pose.md) for producing the JSON.

!!! tip "Batch multiple videos"
    Use `Pose3D.predict_batch(json_paths, vid_paths, out_paths=..., **kwargs)` to
    lift several videos with one estimator; it returns lists of `frames_data` and
    `(N, 17, 3)` arrays.

!!! warning "Rendering pulls in heavy deps"
    `render_video=True` imports optional 3D-rendering dependencies at call time.
    Set `render_video=False` (and/or `save_npy=False`) if you only need the
    returned arrays.

## See also

- [`Pose3D`][physiotrack.pose.pose3D.Pose3D] — the lifting estimator.
- [`PoseCanonicalizer`][physiotrack.PoseCanonicalizer] · [`canonicalize_pose`][physiotrack.canonicalize_pose] — canonicalization API.
- [Pose post-processing API](../api/pose-postprocessing.md) — canonicalizer + evaluation metrics reference.
- [2D Pose guide](pose.md) — produces the 2D keypoints/JSON this consumes.
- [Model Zoo](../model-zoo.md) — MotionBERT, DDHPose, and 3DPCNet weights.
