# Detection

Object detection locates subjects in a frame and returns axis-aligned bounding
boxes with a confidence and class label. In Physiotrack it is the entry point for
most pipelines — person boxes feed pose and tracking, face boxes feed face
parsing and head-orientation. Use it whenever you need *where* something is before
deciding *what* to measure.

Detectors are exposed as ready-to-use presets on [`Detection`][physiotrack.Detection].
Instantiate a preset, call [`predict`][physiotrack.Detection] (or call the instance
directly), and read the returned [`Result`][physiotrack.Result].

## Quick start

```python
from physiotrack import Detection
import cv2

image = cv2.imread("frame_1.png")

detector = Detection.Person(conf=0.25, iou=0.45)   # person boxes (COCO class 0)
result = detector.predict(image)                    # or: detector(image)

boxes = result.boxes            # (N, 4) array of [x1, y1, x2, y2]
annotated = result.plot()       # BGR image with boxes drawn
cv2.imwrite("out.png", annotated)
```

## Available presets

Each preset pins a validated backbone; construction fails fast if you pass an
incompatible model. See the [Model Zoo](../model-zoo.md) for every variant.

| Preset | Backend | Classes | Description |
| --- | --- | --- | --- |
| [`Detection.Person`][physiotrack.Detection.Person] | YOLO | `[0]` (person) | People only; class filter is pinned to person. |
| [`Detection.Face`][physiotrack.Detection.Face] | YOLO | face | Face bounding boxes. |
| [`Detection.VR`][physiotrack.Detection.VR] | YOLO | VR objects | VR-headset object detection. |
| [`Detection.VRStudent`][physiotrack.Detection.VRStudent] | YOLO | VR student | VR-student detection. |
| [`Detection.Custom`][physiotrack.Detection.Custom] | YOLO | any | Run any validated `Models.Detection.*` variant. |

### Two face-detector entry points

PhysioTrack also exposes a top-level [`Face`][physiotrack.Face] preset. Both face
entry points use `Models.Detection.YOLO.FACE.m_face` by default and return the same
box/confidence structure; the difference is the semantic task label:

| Entry point | Result task | Prefer it when… |
| --- | --- | --- |
| `pt.Face()` | `"face"` | face boxes feed head orientation, face parsing, rPPG, or face tracking |
| `pt.Detection.Face()` | `"detect"` | face boxes are one detector choice inside a generic object-detection pipeline |

The dedicated [face examples](face-examples.md) use `pt.Face()` so serialized output
clearly identifies a facial task.

```python
from physiotrack import Detection, Models

# Custom preset: choose an explicit validated model
det = Detection.Custom(model=Models.Detection.YOLO.VR.m_vr, conf=0.3)
```

!!! note "Auto-download"
    On first use a preset's weights are pulled from Hugging Face and cached
    locally; later runs load from disk.

## Key options

Set defaults at construction; override any of `conf` / `iou` / `classes` per call.

| Option | Where | Default | Meaning |
| --- | --- | --- | --- |
| `conf` | constructor + `predict` | `0.25` | Objectness confidence threshold in `[0, 1]`. |
| `iou` | constructor + `predict` | `0.45` | NMS / IoU threshold in `[0, 1]`. |
| `classes` | constructor + `predict` | `None` | Restrict to these class ids (e.g. `[0]`). |
| `device` | constructor | `'cpu'` | `'cpu'`, `'cuda'`, `'mps'`, or an index like `0`. |
| `verbose` | constructor | `False` | Print backend inference logs. |

```python
detector = Detection.Person(device=0)          # run on the first CUDA device
# per-call overrides apply to this call only:
result = detector.predict(image, conf=0.5, iou=0.6)
```

See [`Detection`][physiotrack.Detection] for the full constructor signature.

## Working with results

`predict` returns a [`Result`][physiotrack.Result] for a single frame. It behaves
like a sequence of [`Instance`][physiotrack.Instance] objects and exposes a
vectorized `boxes` view.

```python
result = detector.predict(image)

len(result)              # number of detections
result.boxes             # (N, 4) float array of [x1, y1, x2, y2]

for inst in result:                      # iterate detections
    x1, y1, x2, y2 = inst.box            # (4,) pixel box
    print(inst.cls, inst.cls_name, inst.confidence)

data = result.to_dict()  # JSON-friendly: {"task": "detect", "instances": [...]}
```

Render an annotated copy with [`Result.plot`][physiotrack.Result.plot] — the source
frame is never modified:

```python
annotated = result.plot(conf=True, color=(0, 0, 255), thickness=2)
```

See [Result objects](../api/results.md) for every field and rendering toggle.

## Recipes

!!! example "Batch inference"
    Pass a list or tuple of frames to run them in one call; you get back a
    `list[Result]`, one per frame.

    ```python
    frames = [cv2.imread(p) for p in ("a.png", "b.png", "c.png")]
    results = detector.predict(frames)     # list[Result], same order as input
    counts = [len(r) for r in results]
    ```

!!! tip "Feed a tracker or pose model"
    A detector's boxes are the standard input to downstream stages. The
    [Tracker](tracking.md) needs an `(N, 6)` NumPy array of
    `[x1, y1, x2, y2, confidence, class]`, not the dictionary returned by
    `result.to_dict()`. Build it from the result instances as shown in the tracking
    guide, or let [Pose](pose.md) auto-detect people for you.

!!! warning "Class ids depend on the model"
    `classes` filters by the backbone's own class map. `Detection.Person` already
    pins `[0]`; for `Detection.Custom` inspect the model's labels before filtering.

## See also

- [`Detection` API reference](../api/detection.md) — full class and preset docs.
- [Result objects](../api/results.md) — `boxes`, `Instance`, `to_dict`, `plot`.
- [Model Zoo](../model-zoo.md) — available detection backbones.
- [Pose guide](pose.md) · [Tracking guide](tracking.md) — common next stages.
