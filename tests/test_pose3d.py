"""The 3D-lifting API contract: array in, result object out.

`Pose3D.predict` used to take two filesystem paths and return a bare 2-tuple, which
broke the two rules the rest of the library follows (predictors take arrays; results
are objects, never tuples). These tests pin the new contract, and the input
normalisation that lets a live `VideoResults` and a round-tripped JSON behave alike.

The backend itself is not exercised here -- loading a lifting checkpoint is a
multi-hundred-megabyte download. A stub estimator stands in for it, so the contract is
tested without the weights.
"""
import numpy as np
import pytest

from physiotrack import CanonicalView, Pose3DResult
from physiotrack.pose.pose3D import Pose3D, _as_coco17_sequence
from physiotrack.pose.utils import coco17_to_h36m, coco17_to_halpe26


def coco_sequence(frames=40, seed=0):
    """A smoothly moving synthetic COCO-17 sequence in pixel coordinates."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(200, 800, size=(17, 2))
    xy = np.stack([base + np.array([i * 2.0, 3 * np.sin(i / 5)]) for i in range(frames)])
    return np.concatenate([xy, np.full((frames, 17, 1), 0.9)], axis=2)


def frame_records(keypoints):
    """The serialized per-frame form that `Video.run().to_json()` writes."""
    return [
        {
            "frame_id": i,
            "timestamp": i / 30,
            "instances": [{
                "id": 1,
                "box": [0, 0, 10, 10],
                "keypoints": [
                    {"id": j, "x": float(keypoints[i, j, 0]),
                     "y": float(keypoints[i, j, 1]),
                     "confidence": float(keypoints[i, j, 2])}
                    for j in range(17)
                ],
            }],
        }
        for i in range(len(keypoints))
    ]


class _StubEstimator:
    """Stands in for a lifting backend; records what it was handed."""

    fps_in = 30

    def __init__(self):
        self.received = None

    def infer(self, source, **kwargs):
        self.received = (np.asarray(source), kwargs)
        return np.zeros((len(source), 17, 3))


def make_lifter(framework="MotionBERT"):
    """A Pose3D whose backend is a stub -- no checkpoint, no download."""
    lifter = Pose3D.__new__(Pose3D)          # bypass __init__, which loads weights
    lifter.pose3d_framework = framework
    lifter.pose3d_estimator = _StubEstimator()
    lifter.minfo = {"path": f"Pose3D.{framework}.stub"}
    lifter.model = type("M", (), {"name": "stub"})()
    lifter.device = "cpu"
    lifter.pixel = False
    lifter.clip_len = 27
    return lifter


class TestInputNormalisation:
    def test_array_passthrough(self):
        kp = coco_sequence(12)
        assert _as_coco17_sequence(kp).shape == (12, 17, 3)

    def test_missing_confidence_channel_is_filled(self):
        out = _as_coco17_sequence(coco_sequence(8)[..., :2])
        assert out.shape == (8, 17, 3)
        assert np.all(out[..., 2] == 1.0)

    def test_wholebody_keypoints_are_truncated_to_the_body(self):
        # A 133-joint whole-body pose must be usable: only the COCO body joints lift.
        wide = np.zeros((6, 133, 3))
        wide[:, :17, :] = coco_sequence(6)
        assert np.allclose(_as_coco17_sequence(wide), coco_sequence(6))

    def test_frame_records_match_the_array_path(self):
        kp = coco_sequence(15)
        assert np.allclose(_as_coco17_sequence(frame_records(kp)), kp)

    def test_frames_without_detections_keep_the_sequence_aligned(self):
        # Dropping empty frames would silently desynchronise the 3D output from the
        # source video, which is far worse than a zero-confidence placeholder.
        kp = coco_sequence(10)
        records = frame_records(kp)
        records[3]["instances"] = []
        out = _as_coco17_sequence(records)
        assert len(out) == 10
        assert np.all(out[3] == 0)

    def test_bad_shape_is_rejected(self):
        with pytest.raises(ValueError, match="COCO order"):
            _as_coco17_sequence(np.zeros((5, 9, 3)))

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="nothing to lift"):
            _as_coco17_sequence([])


class TestConverters:
    def test_halpe_conversion_is_consistent_with_the_backend_mapping(self):
        # The synthesised Halpe joints must land where MotionBERT's own halpe2h36m
        # expects them, or every lifted pose is silently wrong.
        from physiotrack.modules.MotionBERT.utils.dataloader import halpe2h36m

        kp = coco_sequence(10)
        h36m = halpe2h36m(coco17_to_halpe26(kp))
        assert np.allclose(h36m[:, 0, :2], (kp[:, 11, :2] + kp[:, 12, :2]) / 2)  # root
        assert np.allclose(h36m[:, 11, :2], kp[:, 5, :2])                        # l shoulder
        assert np.allclose(h36m[:, 9, :2], kp[:, 0, :2])                         # nose

    def test_halpe_first_seventeen_joints_are_untouched(self):
        kp = coco_sequence(7)
        assert np.array_equal(coco17_to_halpe26(kp)[:, :17, :], kp)

    def test_synthesised_joint_confidence_is_the_weakest_parent(self):
        kp = coco_sequence(4)
        kp[:, 11, 2] = 0.1        # one hip is unreliable
        kp[:, 12, 2] = 0.9
        assert np.allclose(coco17_to_halpe26(kp)[:, 19, 2], 0.1)

    def test_h36m_conversion_drops_confidence(self):
        out = coco17_to_h36m(coco_sequence(5))
        assert out.shape == (5, 17, 2)

    def test_h36m_spine_is_between_root_and_thorax(self):
        out = coco17_to_h36m(coco_sequence(6))
        assert np.allclose(out[:, 7], (out[:, 0] + out[:, 8]) / 2)


class TestPredictContract:
    def test_returns_a_result_object_not_a_tuple(self):
        result = make_lifter().predict(coco_sequence(30), fps=30)
        assert isinstance(result, Pose3DResult)
        assert not isinstance(result, tuple)

    def test_result_carries_shape_fps_and_metadata(self):
        result = make_lifter().predict(coco_sequence(30), fps=25)
        assert result.poses.shape == (30, 17, 3)
        assert result.fps == 25
        assert result.meta.device == "cpu"
        assert result.meta.units == {"poses": "relative"}

    def test_frame_count_is_preserved(self):
        for n in (5, 30, 64):
            assert len(make_lifter().predict(coco_sequence(n), fps=30)) == n

    def test_motionbert_receives_halpe_layout(self):
        lifter = make_lifter("MotionBERT")
        lifter.predict(coco_sequence(20), fps=30)
        handed, _ = lifter.pose3d_estimator.received
        assert handed.shape == (20, 26, 3), "MotionBERT expects Halpe-26"

    def test_ddh_receives_h36m_layout(self):
        lifter = make_lifter("DDH")
        lifter.predict(coco_sequence(20), fps=30, frame_size=(1920, 1080))
        handed, _ = lifter.pose3d_estimator.received
        assert handed.shape == (20, 17, 2), "DDHPose expects H3.6M (x, y)"

    def test_ddh_without_frame_size_fails_loudly(self):
        # DDHPose normalises pixel coordinates; guessing a frame size would silently
        # scale every pose wrongly.
        with pytest.raises(ValueError, match="frame_size"):
            make_lifter("DDH").predict(coco_sequence(20), fps=30)

    def test_motionbert_does_not_require_frame_size(self):
        assert make_lifter("MotionBERT").predict(coco_sequence(20), fps=30) is not None

    def test_canonical_view_is_recorded_on_the_result(self):
        from physiotrack import Models

        result = make_lifter().predict(
            coco_sequence(30), fps=30,
            canonical_view=CanonicalView.FRONT,
            canonical_model=Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
        )
        assert result.to_dict()["view"] == "FRONT"

    def test_video_results_input_is_accepted(self):
        from physiotrack import FrameResult, Result, ResultMeta, VideoResults

        kp = coco_sequence(12)
        frames = VideoResults(
            FrameResult(
                result=Result.from_dict({"task": "pose", **rec}, orig_img=None),
                meta=ResultMeta(frame_index=i, timestamp=i / 30, fps=30),
            )
            for i, rec in enumerate(frame_records(kp))
        )
        result = make_lifter().predict(frames, fps=30)
        assert len(result) == 12
