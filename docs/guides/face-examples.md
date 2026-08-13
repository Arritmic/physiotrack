# Face Detection & Tracking Examples

PhysioTrack includes two runnable, self-contained examples with small synthetic
inputs. They are designed to answer three practical questions: does the model run,
what did it return, and how can those results be inspected outside Python?

| Example | Input | Main outputs |
| --- | --- | --- |
| [`examples/face_detection`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_detection) | four scene images | annotated PNGs, per-image JSON, `summary.csv`, `run.json` |
| [`examples/face_tracking`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_tracking) | one 10-second clip | annotated MP4, per-frame JSONL, track CSV, summary JSON |

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

For a quick smoke check, process only the first 120 frames; add `--show` only when
a desktop window is available:

```bash
python examples/face_tracking/track_faces.py --device cuda --max-frames 120
python examples/face_tracking/track_faces.py --input path/to/video.mp4 --show
```

This is **tracking by detection**. For each frame the script:

1. calls [`Face.predict`][physiotrack.Face.predict];
2. converts every face to a row `[x1, y1, x2, y2, confidence, class]`;
3. advances one stateful [`Tracker`][physiotrack.Tracker];
4. draws track IDs/trails and the model/count panel;
5. writes visual, structured, and tabular results.

The face count and active-track count can differ. A new detector box may need a few
frames before a tracker reports it, and a tracker may temporarily retain an object
through a missed detection.

### Tracking output

| Output | Contents |
| --- | --- |
| `*_tracked.mp4` | boxes, temporary IDs, trails, per-frame counts, detector model and tracker; source audio is not copied |
| `*_frames.jsonl` | one JSON object per frame, safe to stream without loading the whole video |
| `*_tracks.csv` | one row per active track per frame: time, ID, box, confidence and class |
| `*_summary.json` | source metadata, configuration, output paths, timings, face-count aggregates and all observed temporary IDs |

One JSONL record has this shape:

```json
{
  "frame_index": 42,
  "timestamp_seconds": 1.75,
  "faces_detected": 2,
  "active_track_ids": [1, 2],
  "face_result": {"task": "face", "instances": []},
  "track_result": {"task": "track", "instances": []}
}
```

The shortened empty lists above only show the schema. Real records include one
serialized instance per detection or active track. Tracking instances additionally
carry an `id`. Both result types deliberately use the same `instances` key.

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
