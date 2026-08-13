# Face Detection & Tracking Examples

PhysioTrack includes two runnable, self-contained examples with small synthetic
inputs. They are designed to answer three practical questions: does the model run,
what did it return, and how can those results be inspected outside Python?

| Example | Input | Main outputs |
| --- | --- | --- |
| [`examples/face_detection`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_detection) | four scene images | annotated PNGs, per-image JSON, `summary.csv`, `run.json` |
| [`examples/face_tracking`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_tracking) | one 10-second clip | annotated MP4, per-frame JSON, track CSV |

The generated `results/` directories are ignored by Git. Run the scripts locally,
inspect their outputs, and commit only deliberate documentation assets—not an entire
inference run.

!!! info "Synthetic example assets"
    The contributor confirms that the bundled visuals were generated through Google
    Gemini on August 13, 2026, with their people and scene imagery created using Nano
    Banana 2 (Gemini 3.1 Flash Image). They depict no real living people and are
    included for research, evaluation, testing, and documentation. The media is
    dedicated under CC0 1.0 Universal for any copyright and related rights held by
    the contributor; code and documentation remain GPL-3.0-or-later.
    See each example's `README.md` and `data/MEDIA.yml` for the complete notice,
    model scope, SynthID status, and checksums.

## Detect faces in images

From the repository root, after installing PhysioTrack:

```bash
python examples/face_detection/detect_faces.py
```

This processes the bundled selfie, point-of-view, crowd, and VR scenes on CPU. The
model is constructed once and reused for every image. Use CUDA, a different model
size, or your own input like this:

```bash
python examples/face_detection/detect_faces.py --device cuda
python examples/face_detection/detect_faces.py --model nano --input path/to/images
python examples/face_detection/detect_faces.py --input path/to/one_image.jpg
```

Each annotated image has normal face boxes and confidence labels plus a top-left
panel containing:

- `Faces detected`: the number of [`Instance`][physiotrack.Instance] objects in
  this image's [`Result`][physiotrack.Result];
- `Detector`: the exact checkpoint filename, such as `yolov11m-face.pt`;
- `Device` and elapsed inference time for this call.

### Detection output

```text
examples/face_detection/results/
├── annotated/<scene>/<image>.png
├── predictions/<scene>/<image>.json
├── summary.csv
└── run.json
```

`summary.csv` has one row per input image:

| Field | Meaning |
| --- | --- |
| `image`, `scene` | relative input path and its first directory (for example `crowd`) |
| `width`, `height` | decoded image dimensions in pixels |
| `faces_detected` | number of boxes retained after confidence filtering and NMS |
| `mean_confidence`, `minimum_confidence` | summaries of the retained face confidences; blank when none were found |
| `inference_ms` | one end-to-end `detector.predict(image)` call; model setup is excluded |
| `model`, `device_requested` | checkpoint filename and requested compute device |
| `status`, `error` | `ok`, or a readable exception while other images continue processing |

Each per-image JSON is the detailed, typed counterpart:

```json
{
  "source": {"image": "selfie/two_person_selfie.jpg", "width": 1643, "height": 2200},
  "configuration": {"entry_point": "physiotrack.Face", "model": "yolov11m-face.pt"},
  "timing": {"inference_ms": 123.4},
  "result": {
    "task": "face",
    "instances": [
      {"box": [100.0, 120.0, 400.0, 520.0], "confidence": 0.97, "cls": 0}
    ]
  }
}
```

The example is the outer experiment record; `result` is exactly
[`Result.to_dict()`][physiotrack.Result.to_dict]. Coordinates are `[x1, y1, x2,
y2]` pixels. `run.json` records the thresholds, warm-up count, model setup time,
aggregate counts/timing, PhysioTrack/OpenCV/PyTorch versions, CUDA visibility, and
GPU name so the run can be interpreted later.

## Track faces in a video

Run the complete bundled clip:

```bash
python examples/face_tracking/track_faces.py
```

Add `--show` only when a desktop window is available:

```bash
python examples/face_tracking/track_faces.py --device cuda
python examples/face_tracking/track_faces.py --input path/to/video.mp4 --show
```

This is **tracking by detection**, and it is plain composition of two predictors
through the core [`Video`][physiotrack.Video] pipeline — the same pattern as
`examples/pose_video.py` and `examples/tracker_aided_pose_video.py`:

```python
detector = pt.Face(model=..., conf=..., iou=..., device=...)
tracker  = pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0]))
video    = pt.Video(source=..., detector=detector, tracker=tracker, output_dir=...)
results  = video.run(output_video, output_json)
```

`Video` runs the face detector per frame, feeds the boxes to the stateful tracker,
draws IDs and trails, writes the annotated video (H.264 when available, MPEG-4
otherwise), and returns one [`FrameResult`][physiotrack.FrameResult] per frame whose
instances carry persistent track `id`s (`task="track"`). The script then derives a
per-track CSV from those results in a few lines.

The face count and active-track count can differ. A new detector box may need a few
frames before a tracker reports it, and a tracker may temporarily retain an object
through a missed detection.

### Tracking output

| Output | Contents |
| --- | --- |
| `*_tracked.mp4` | boxes, temporary IDs, and trails; source audio is not copied |
| `*_result.json` | the serialized [`VideoResults`][physiotrack.VideoResults] — one record per frame with `frame_id`, `timestamp`, and tracked `instances` |
| `*_tracks.csv` | one row per active track per frame: time, ID, box, confidence and class |

One JSON frame record has this shape:

```json
{
  "frame_id": 42,
  "timestamp": 1.75,
  "instances": [
    {"box": [100.0, 120.0, 180.0, 220.0], "confidence": 0.94, "cls": 0, "id": 1}
  ]
}
```

This is the same per-frame schema every `Video` pipeline produces (see the
[Video guide](video.md)), so the output feeds any tooling that already consumes
PhysioTrack results.

!!! warning "A track ID is not identity recognition"
    An ID is a temporary association inside one tracker run. It can change after a
    long occlusion, exit/re-entry, or association error. It is neither a name nor a
    biometric identity and must not be described as face recognition.

## How to interpret the examples

The bundled media is synthetic and carries no ground-truth boxes or tracks. A missed
face, extra box, or ID switch is useful qualitative evidence for debugging, but it is
not enough to calculate accuracy. Use a labelled benchmark and a fixed evaluation
protocol for claims about precision, recall, average precision, or tracking metrics;
the [face validation guide](face-validation.md) explains the separation.

## API choice: `Face()` or `Detection.Face()`?

The examples intentionally use top-level [`Face`][physiotrack.Face]. It returns
`Result(task="face")` and communicates that the boxes feed a facial pipeline.
[`Detection.Face`][physiotrack.Detection.Face] uses the same default YOLO face
checkpoint but belongs to the generic detector namespace and returns
`Result(task="detect")`. Box coordinates and confidences follow the same contract;
choose the entry point whose task semantics fit the pipeline.
