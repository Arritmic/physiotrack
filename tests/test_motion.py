"""Motion keypoint extraction and joint-angle unit consistency."""
import math

import numpy as np
import pandas as pd
import pytest

from physiotrack.signals import (
    extract_keypoint_sequence_2d, extract_keypoint_sequence_3d, add_pelvic_centroid,
    compute_all_joint_angles, joint_angles, compute_rom_angles,
)
from physiotrack.signals.motion.features import (
    compute_joint_angle_2d, compute_joint_angle_3d,
)


def _frame_with_hips():
    return [{
        "frame_id": 0, "timestamp": 0.0,
        "instances": [{
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
    df = extract_keypoint_sequence_2d(data, keypoint_id=135)
    assert len(df) == 1
    assert df.iloc[0]["x"] == 20.0 and df.iloc[0]["y"] == 20.0


def test_extract_pelvic_centroid_3d_no_keyerror():
    data = add_pelvic_centroid(_frame_with_hips(), "coco_wholebody")
    df = extract_keypoint_sequence_3d(data, keypoint_id=135)
    assert len(df) == 1
    assert df.iloc[0]["x"] == 2.0 and df.iloc[0]["z"] == 2.0


# --- Joint angles are degrees everywhere ------------------------------------------
# Regression guard: compute_joint_angle_2d/3d once returned radians while
# joint_angles() converted to degrees and compute_all_joint_angles() did not, so the
# two public paths disagreed by a factor of 180/pi on identical geometry.

@pytest.mark.parametrize("fn, a, b, c, expected", [
    (compute_joint_angle_2d, (1, 0), (0, 0), (0, 1), 90.0),
    (compute_joint_angle_2d, (-1, 0), (0, 0), (1, 0), 180.0),
    (compute_joint_angle_2d, (1, 0), (0, 0), (1, 1), 45.0),
    (compute_joint_angle_3d, (1, 0, 0), (0, 0, 0), (0, 1, 0), 90.0),
    (compute_joint_angle_3d, (-1, 0, 0), (0, 0, 0), (1, 0, 0), 180.0),
])
def test_interior_angle_returns_degrees(fn, a, b, c, expected):
    assert fn(a, b, c) == pytest.approx(expected)


@pytest.mark.parametrize("fn, a, b, c", [
    (compute_joint_angle_2d, (0, 0), (0, 0), (1, 0)),
    (compute_joint_angle_3d, (0, 0, 0), (0, 0, 0), (1, 0, 0)),
])
def test_interior_angle_degenerate_segment_is_nan(fn, a, b, c):
    assert math.isnan(fn(a, b, c))


def test_joint_angles_and_dataframe_path_agree_in_degrees():
    # A right angle at the left elbow (shoulder 5, elbow 7, wrist 9).
    kps = [
        {"id": 5, "x": 0.0, "y": 0.0, "confidence": 1.0},
        {"id": 7, "x": 1.0, "y": 0.0, "confidence": 1.0},
        {"id": 9, "x": 1.0, "y": 1.0, "confidence": 1.0},
    ]
    per_frame = joint_angles(kps, joints=["leftElbow"])["leftElbow"]

    wide = pd.DataFrame([{"5_x": 0.0, "5_y": 0.0, "7_x": 1.0, "7_y": 0.0,
                         "9_x": 1.0, "9_y": 1.0}])
    sequence = compute_all_joint_angles(wide)["ang_2d_leftElbow"].iloc[0]

    assert per_frame == pytest.approx(90.0)
    assert sequence == pytest.approx(90.0)
    assert per_frame == pytest.approx(sequence)


# --- signals accept the predictor result objects, not only serialized dicts --------

def _right_angle_elbow_parts():
    """A right angle at the left elbow (COCO ids 5 shoulder, 7 elbow, 9 wrist)."""
    import physiotrack as pt

    dicts = [
        {"id": 5, "x": 0.0, "y": 0.0, "confidence": 1.0},
        {"id": 7, "x": 1.0, "y": 0.0, "confidence": 1.0},
        {"id": 9, "x": 1.0, "y": 1.0, "confidence": 1.0},
    ]
    keypoints = pt.Keypoints(dicts, "COCO")
    instance = pt.Instance(id=1, box=np.array([0, 0, 10, 10], np.float32),
                           keypoints=keypoints)
    result = pt.Result(orig_img=np.zeros((32, 32, 3), np.uint8), instances=[instance],
                       task="pose", architecture="COCO")
    return dicts, keypoints, instance, result


def test_joint_angles_accepts_result_objects_and_dicts_alike():
    dicts, keypoints, instance, result = _right_angle_elbow_parts()
    for source in (dicts, keypoints, instance, result):
        got = joint_angles(source, joints=["leftElbow"])
        assert got["leftElbow"] == pytest.approx(90.0), f"failed for {type(source).__name__}"


def test_joint_angles_rejects_ambiguous_multi_instance_result():
    import physiotrack as pt

    _, _, instance, _ = _right_angle_elbow_parts()
    two = pt.Result(orig_img=np.zeros((32, 32, 3), np.uint8),
                    instances=[instance, instance], task="pose", architecture="COCO")
    # Choosing a subject implicitly is exactly the class of bug this guards against.
    with pytest.raises(ValueError, match="one subject"):
        joint_angles(two, joints=["leftElbow"])


def test_joint_angles_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Expected Keypoints"):
        joint_angles(42)


def test_rom_angles_accepts_an_instance():
    _, _, instance, _ = _right_angle_elbow_parts()
    # No hip keypoints present, so the result is empty -- but it must not raise.
    assert compute_rom_angles(instance) == {}


def test_rom_angles_are_degrees_and_use_neutral_offset():
    # Thigh (hip 11 -> knee 13) straight down, trunk reference at shoulder 5 straight
    # up: the raw hip angle is 180 deg, so flexion reads scale*180 + 180 == 0 at neutral.
    kps = [
        {"id": 11, "x": 0.0, "y": 0.0, "confidence": 1.0},
        {"id": 5, "x": 0.0, "y": -1.0, "confidence": 1.0},
        {"id": 13, "x": 0.0, "y": 1.0, "confidence": 1.0},
    ]
    out = compute_rom_angles(kps, movements=["leftHipFlexion"])
    assert out["leftHipFlexion"] == pytest.approx(0.0, abs=1e-6)
