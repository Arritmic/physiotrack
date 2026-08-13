"""Fast tests for the public face example glue; no model weights are loaded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import physiotrack as pt


ROOT = Path(__file__).resolve().parents[1]


def load_example(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_face_tracking_example_converts_results_to_tracker_rows():
    example = load_example("examples/face_tracking/track_faces.py")
    result = pt.Result(
        orig_img=np.zeros((8, 8, 3), dtype=np.uint8),
        task="face",
        instances=[
            pt.Instance(
                box=np.array([1, 2, 7, 8], dtype=np.float32),
                confidence=0.75,
                cls=0,
                cls_name="face",
            )
        ],
    )

    rows = example.result_to_tracker_rows(result)

    assert rows.shape == (1, 6)
    assert rows.dtype == np.float32
    assert np.allclose(rows[0], [1, 2, 7, 8, 0.75, 0])
    assert example.result_to_tracker_rows(
        pt.Result(orig_img=result.orig_img, task="face", instances=[])
    ).shape == (0, 6)


def test_information_panel_changes_only_the_corner_region():
    example = load_example("examples/face_detection/detect_faces.py")
    image = np.full((480, 640, 3), 200, dtype=np.uint8)

    annotated = example.add_info_panel(image, ["Faces detected: 2", "Detector: test.pt"])

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated[:100, :300], image[:100, :300])
    assert np.array_equal(annotated[-20:, -20:], image[-20:, -20:])

