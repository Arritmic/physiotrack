# Face & Head Orientation

Detect faces and estimate each face's 3D **head pose** — yaw, pitch, and roll.
Physiotrack pairs a lightweight face detector ([`Face`][physiotrack.Face] or the
VR-tuned [`VRFace`][physiotrack.VRFace]) with a head-pose estimator
([`FaceOrientation`][physiotrack.FaceOrientation], 6DRepNet360). Use it for gaze
proxies, attention/engagement analysis, or head-motion tracking — including
subjects wearing head-mounted displays.

Detection returns a [`Result`][physiotrack.Result] with `task="face"` whose
instances carry face boxes; orientation adds an `orientation` dict per
[`Instance`][physiotrack.Instance], and `result.plot()` draws the pose axes.

## Quick start

```python
import cv2
import physiotrack as pt

face_detector   = pt.VRFace(device=0)
face_orientation = pt.FaceOrientation(model=pt.Models.Pose3D.FaceOrientation.VR, device=0)

img = cv2.imread("kinect_s1_v1_frame1.png")

# Detect faces, then estimate head pose from the boxes
boxes  = face_detector.predict(img).boxes          # (N, 4) ndarray
result = face_orientation.predict(img, boxes)       # Result(task="face")

for inst in result:
    print(inst.orientation)                         # {"yaw":.., "pitch":.., "roll":..}

cv2.imwrite("face_orientation_output.png", result.plot())   # draws pose axes
```

## Available presets

### Face detectors

| Preset | Backend model | Description |
| --- | --- | --- |
| [`Face`][physiotrack.Face] | `Models.Detection.YOLO.FACE.m_face` | General-purpose YOLO face detector. |
| [`VRFace`][physiotrack.VRFace] | `Models.Detection.YOLO.VRFACE.l_vrface` | YOLOv12l-face tuned for VR headsets; robust to HMD occlusion. |

Both take the standard detector arguments (`conf=0.25`, `iou=0.45`, `classes`,
`device`, `verbose`), return a `Result` with `task="face"`, and can be called
directly (`face(img)`) as an alias for `predict`.

### Head-pose estimator

| Model enum | Description |
| --- | --- |
| `Models.Pose3D.FaceOrientation.default` | General 6DRepNet360 (300W-LP + Panoptic). Used when `model=None`. |
| `Models.Pose3D.FaceOrientation.VR` | VR-tuned variant for headset-wearing subjects. |

See the [Model Zoo](../model-zoo.md) for weights and downloads.

## Key options

[`FaceOrientation`][physiotrack.FaceOrientation] constructor:

| Argument | Default | Purpose |
| --- | --- | --- |
| `model` | `None` → `FaceOrientation.default` | Orientation model variant. |
| `device` | `'cpu'` | `'cpu'`, `'cuda'`, or a device id like `0`. |
| `verbose` | `False` | Print progress / download notes. |

`predict(source, bboxes=None)` takes a BGR frame `(H, W, 3)`, a path to an image
file, or a list of either, plus face boxes `(N, 4)` as `[x1, y1, x2, y2]`. If
`bboxes` is omitted the **whole image is treated as a single face**, so pass boxes
from a detector for multi-face frames. Pass a list of frames (and a matching list of
box arrays) for batch inference; the return is then a list of results, one per frame.
Calling the instance (`orient(frame, boxes)`) is an alias for `predict`.

!!! note "The `orientation` dict"
    Each returned `Instance` has `inst.orientation` = `{"yaw", "pitch", "roll"}`
    in **degrees**. Other instance fields follow the shared
    [`Instance`][physiotrack.Instance] contract — `inst.box` holds the face box.

## Working with results

```python
result = face_orientation.predict(img, boxes)

for inst in result:                 # each face is an Instance
    yaw   = inst.orientation["yaw"]
    pitch = inst.orientation["pitch"]
    roll  = inst.orientation["roll"]
    x1, y1, x2, y2 = inst.box       # (4,) face box

result.boxes        # (N, 4) all face boxes
result.to_dict()    # JSON-friendly dict; orientation stays under "orientation"
```

See [Result objects](../api/results.md) for the full container API.

### Drawing the pose axis

`result.plot()` renders the yaw/pitch/roll axes automatically. For manual control
you can call the drawing helpers directly:

```python
from physiotrack.face import draw_axis, plot_pose_cube

vis = img.copy()
for inst in result:
    pose = inst.orientation
    x1, y1, x2, y2 = inst.box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    size = max(x2 - x1, y2 - y1) * 0.6

    vis = draw_axis(vis, yaw=pose["yaw"], pitch=pose["pitch"], roll=pose["roll"],
                    tdx=cx, tdy=cy, size=size)
    # or plot_pose_cube(...) for a projected cube instead of axes

cv2.imwrite("face_orientation_output_manual.png", vis)
```

## Recipes & tips

!!! tip "Reuse detector boxes"
    Detect once, orient many: pass the detector's `.boxes` straight into
    `face_orientation.predict(img, boxes)` rather than re-running detection.

!!! warning "First run downloads weights"
    On first use the face and orientation weights auto-download (from Hugging
    Face, with a 6DRepNet source fallback for the orientation model).

!!! info "Face parsing / skin segmentation lives elsewhere"
    Per-pixel face **parsing** (skin, eyes, lips, etc.) and skin-based
    physiological signals such as rPPG heart rate are covered in the
    [Signals guide](signals.md), not here.

## Runnable detection and tracking examples

The repository includes synthetic media and command-line examples that run without
editing file paths. The image example saves annotated PNGs, per-image JSON, a CSV
summary and run metadata. The video example runs face detection and tracking
through the core [`Video`][physiotrack.Video] pipeline and saves an annotated MP4,
per-frame JSON, and a per-track CSV.

See [Face Detection & Tracking Examples](face-examples.md) for commands, output
schemas, and interpretation. These are qualitative teaching examples; use the
[Face Detection & Tracking Validation](face-validation.md) guide before making
accuracy claims.

## See also

- [`Face`][physiotrack.Face] · [`VRFace`][physiotrack.VRFace] · [`FaceOrientation`][physiotrack.FaceOrientation] — API reference.
- [Face API page](../api/face.md) — full parameter tables.
- [Result objects](../api/results.md) — `Result`, `Instance`, the `orientation` dict.
- [Signals guide](signals.md) — face parsing and rPPG heart rate.
- [Model Zoo](../model-zoo.md) — face detector and head-pose weights.
- [Face examples](face-examples.md) — runnable image/video demos and output schemas.
- [Face validation](face-validation.md) — datasets, manifests, and reporting boundaries.
