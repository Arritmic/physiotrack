"""Fast tests for the face examples and the Video glue they rely on; no model
weights are loaded — detection is stubbed where a detector is needed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import physiotrack as pt
from physiotrack.core.overlay import draw_info_panel


ROOT = Path(__file__).resolve().parents[1]
TRACKING_CLIP = ROOT / "examples" / "face_tracking" / "data" / "students_face_tracking.mp4"


def load_example(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Rows:
    """Minimal stand-in for a tensor: .cpu().numpy() -> (N, 6) rows."""

    def __init__(self, rows: np.ndarray):
        self._rows = rows

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._rows


class _StubBoxes:
    def __init__(self, rows: np.ndarray):
        self.data = _Rows(rows)


class _StubYoloResult:
    def __init__(self, rows: np.ndarray):
        self.boxes = _StubBoxes(rows)


class _StubDetector:
    """Returns two fixed face boxes every frame, mimicking the YOLO detect API.

    Deliberately has no ``detect_batch`` so Video exercises its per-frame path.
    """

    def detect(self, frame, **kwargs):
        rows = np.array(
            [
                [40.0, 40.0, 120.0, 140.0, 0.9, 0.0],
                [200.0, 60.0, 280.0, 160.0, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        return [_StubYoloResult(rows)], frame


def test_face_detection_example_finds_the_bundled_scene_images():
    example = load_example("examples/face_detection/detect_faces.py")
    root, paths = example.image_paths(example.DEFAULT_INPUT)

    assert root == example.DEFAULT_INPUT.resolve()
    assert [path.relative_to(root).as_posix() for path in paths] == [
        "crowd/outdoor_cafe_crowd.jpg",
        "pov/exercise_class_pov.jpg",
        "selfie/two_person_selfie.jpg",
        "vr/vr_training_lab.jpg",
    ]


def test_cpu_gpu_example_matches_same_boxes_and_empty_results():
    example = load_example("examples/face_detection/compare_cpu_gpu.py")

    def result(boxes, confidences):
        return pt.Result(
            orig_img=np.zeros((8, 8, 3), dtype=np.uint8),
            task="face",
            instances=[
                pt.Instance(box=np.array(box, dtype=np.float32), confidence=confidence)
                for box, confidence in zip(boxes, confidences)
            ],
        )

    cpu = result([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8])
    gpu = result([[1, 1, 11, 11], [50, 50, 60, 60]], [0.85, 0.7])
    agreement = example.match_predictions(cpu, gpu, minimum_iou=0.5)

    assert agreement["matched_detections"] == 1
    assert agreement["unmatched_cpu"] == 1
    assert agreement["unmatched_gpu"] == 1
    assert agreement["matches"][0]["absolute_confidence_difference"] == pytest.approx(0.05)
    assert example.box_iou_matrix(np.empty((0, 4)), np.empty((0, 4))).shape == (0, 0)


def test_vr_detection_example_defines_distinct_detector_questions():
    example = load_example("examples/vr_detection/detect_vr_people.py")

    assert set(example.DETECTORS) == {"vr_head", "vr_person", "person"}
    assert pt.Models.Detection.YOLO.VR.m_vr.value == "yolo11m_VR_head.pt"
    assert "l_vr" not in pt.Models.Detection.YOLO.VR.__members__
    assert example.DEFAULT_INPUT == (
        ROOT / "examples/face_detection/data/vr/vr_training_lab.jpg"
    )
    assert example.select_models(["vr_head"], "medium") == {
        "vr_head": pt.Models.Detection.YOLO.VR.m_vr
    }
    largest = example.select_models(list(example.DETECTORS), "largest")
    assert largest == {
        "vr_head": pt.Models.Detection.YOLO.VR.m_vr,
        "vr_person": pt.Models.Detection.YOLO.VRSTUDENT.l_vrstudent,
        "person": pt.Models.Detection.YOLO.PERSON.l_person,
    }
    with pytest.raises(ValueError, match="No large checkpoint.*vr_head"):
        example.select_models(["vr_head"], "large")
    result = pt.Result(
        orig_img=np.zeros((8, 8, 3), dtype=np.uint8),
        task="detect",
        instances=[
            pt.Instance(cls=0, cls_name="VR-person"),
            pt.Instance(cls=0, cls_name="VR-person"),
        ],
    )

    assert example.class_counts(result) == {"VR-person": 2}
    lines = example.panel_lines("vr_person", result, 12.5, "test.pt", "cpu")
    assert "VR-person: 2" in lines
    assert "Detector: test.pt" in lines

    first = np.zeros((20, 40, 3), dtype=np.uint8)
    second = np.zeros((10, 20, 3), dtype=np.uint8)
    stacked = example.stack_views([first, second], width=20)
    assert stacked.shape == (20, 20, 3)


def test_video_exports_tracked_instances_without_a_pose_estimator():
    """A detector + tracker pipeline must export per-frame tracked instances.

    This is the core contract the face-tracking example builds on: without a pose
    estimator, Video previously returned empty frames, which forced examples to
    re-implement the capture/track/export loop by hand.
    """
    tracker = pt.Tracker(pt.TrackerConfig(
        tracker_type="ocsort", classes=[0], enable_subject_lock=False,
    ))
    video = pt.Video(
        source=TRACKING_CLIP,
        detector=_StubDetector(),
        tracker=tracker,
        fps=5,  # subsample so the test stays fast
    )
    results = video.run()

    assert len(results) > 0
    tracked = [inst for frame in results for inst in frame if inst.id is not None]
    assert tracked, "expected at least one instance with a persistent track id"
    assert all(inst.box is not None and len(inst.box) == 4 for inst in tracked)
    assert results[-1].result.task == "track"

    serialized = results[-1].to_dict()
    assert "instances" in serialized
    assert any("id" in inst for inst in serialized["instances"])


def test_tracking_example_derives_csv_rows_from_video_results(tmp_path):
    example = load_example("examples/face_tracking/track_faces.py")
    meta = pt.ResultMeta(frame_index=3, timestamp=0.1, fps=30.0)
    frame = pt.FrameResult(
        result=pt.Result(
            orig_img=np.zeros((8, 8, 3), dtype=np.uint8),
            task="track",
            instances=[
                pt.Instance(box=np.array([1, 2, 7, 8], dtype=np.float32),
                            confidence=0.75, cls=0, id=5),
                pt.Instance(box=None, id=9),         # skipped: no box
                pt.Instance(box=np.array([0, 0, 4, 4], dtype=np.float32)),  # skipped: no id
            ],
            meta=meta,
        ),
        meta=meta,
    )

    csv_path = tmp_path / "tracks.csv"
    ids = example.write_tracks_csv([frame], csv_path)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()

    assert ids == {5}
    assert len(lines) == 2  # header + the one instance with both id and box
    assert lines[1].startswith("3,0.1,5,1.0,2.0,7.0,8.0,0.75,0")


def test_information_panel_changes_only_the_requested_corner():
    image = np.full((480, 640, 3), 200, dtype=np.uint8)

    annotated = draw_info_panel(image, ["Faces detected: 2", "Detector: test.pt"])

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated[:60, :200], image[:60, :200])
    assert np.array_equal(annotated[-20:, -20:], image[-20:, -20:])
    assert np.array_equal(image, np.full((480, 640, 3), 200, dtype=np.uint8))
