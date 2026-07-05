"""Motion keypoint extraction: pelvic-centroid regression (KeyError on id 135)."""
from physiotrack.signals import (
    extract_keypoint_sequence_2d, extract_keypoint_sequence_3d, add_pelvic_centroid,
)


def _frame_with_hips():
    return [{
        "frame_id": 0, "timestamp": 0.0,
        "detections": [{
            "id": 1,
            "keypoints": [
                {"id": 11, "x": 10.0, "y": 20.0, "confidence": 0.9},
                {"id": 12, "x": 30.0, "y": 20.0, "confidence": 0.9},
            ],
            "keypoints3D": [
                {"id": 11, "x": 1.0, "y": 2.0, "z": 3.0},
                {"id": 12, "x": 3.0, "y": 2.0, "z": 1.0},
            ],
        }],
    }]


def test_extract_pelvic_centroid_2d_no_keyerror():
    data = add_pelvic_centroid(_frame_with_hips(), "coco_wholebody")
    df = extract_keypoint_sequence_2d(data, keypoint_id=135, original_fps=30.0)
    assert len(df) == 1
    assert df.iloc[0]["x"] == 20.0 and df.iloc[0]["y"] == 20.0


def test_extract_pelvic_centroid_3d_no_keyerror():
    data = add_pelvic_centroid(_frame_with_hips(), "coco_wholebody")
    df = extract_keypoint_sequence_3d(data, keypoint_id=135, original_fps=30.0)
    assert len(df) == 1
    assert df.iloc[0]["x"] == 2.0 and df.iloc[0]["z"] == 2.0
