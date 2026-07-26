"""Geometric canonicalization: one derivation, four views.

The back/left/right transforms used to be three copies of the same forty lines, differing
only in a 3x3 rotation. They now share one code path. These tests pin the *geometric*
properties that made the duplication safe to remove, so a future change to the shared
path cannot quietly alter one view's behaviour.

The learned 3DPCNet path is not covered here: it needs a downloaded checkpoint.
"""
import numpy as np
import pytest

import physiotrack as pt
from physiotrack.pose.canonicalizer import CanonicalView, PoseCanonicalizer


@pytest.fixture
def poses():
    """A synthetic 3D pose sequence in H3.6M joint order."""
    return np.random.default_rng(7).normal(size=(25, 17, 3)) * 100


class TestViewGeometry:
    def test_all_four_views_preserve_shape(self, poses):
        for view in CanonicalView:
            assert PoseCanonicalizer.to_canonical_geometric(poses, view).shape == poses.shape

    @pytest.mark.parametrize("view", list(CanonicalView))
    def test_rotations_are_proper_and_orthonormal(self, poses, view):
        # A canonicalizing transform must be a rigid rotation: orthonormal with
        # determinant +1. A determinant of -1 would mirror the body, silently swapping
        # left and right limbs.
        #
        # The tolerance is 1e-6, not machine epsilon: building R_front from normalised
        # cross products accumulates ~2e-7 of float64 error. That is physically
        # irrelevant, while the failures this test exists to catch -- a mirrored or
        # non-rigid transform -- are O(1).
        _, r = PoseCanonicalizer.to_canonical_geometric(poses, view, return_rotation=True)
        assert r.shape == (len(poses), 3, 3)
        eye = np.einsum("nij,nkj->nik", r, r)
        assert np.allclose(eye, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(r), 1.0, atol=1e-6)

    @pytest.mark.parametrize("view", list(CanonicalView))
    def test_bone_lengths_are_unchanged(self, poses, view):
        # Rotation is rigid, so every inter-joint distance must survive it. This is the
        # single strongest check that a view transform has not distorted the skeleton.
        out = PoseCanonicalizer.to_canonical_geometric(poses, view)
        before = np.linalg.norm(poses[:, 1:] - poses[:, :1], axis=-1)
        after = np.linalg.norm(out[:, 1:] - out[:, :1], axis=-1)
        assert np.allclose(before, after, atol=1e-6)

    def test_front_view_is_the_base_of_the_others(self, poses):
        # The three derived views are the front view plus one fixed rotation; each must
        # therefore differ from front, and from each other.
        views = {v: PoseCanonicalizer.to_canonical_geometric(poses, v)
                 for v in CanonicalView}
        for v in (CanonicalView.BACK, CanonicalView.LEFT_SIDE, CanonicalView.RIGHT_SIDE):
            assert not np.allclose(views[v], views[CanonicalView.FRONT])
        assert not np.allclose(views[CanonicalView.LEFT_SIDE],
                               views[CanonicalView.RIGHT_SIDE])

    def test_back_view_is_a_half_turn_from_front(self, poses):
        # Applying the back rotation twice returns to the front view, which is what
        # "180 degrees" means and would fail if the matrix were ever mistyped.
        front = PoseCanonicalizer.to_canonical_geometric(poses, CanonicalView.FRONT)
        back = PoseCanonicalizer.to_canonical_geometric(poses, CanonicalView.BACK)
        r = PoseCanonicalizer._VIEW_ROTATIONS[CanonicalView.BACK]
        centre, _, _ = PoseCanonicalizer.extract_torso_plane(back)
        again = np.matmul(back - centre[:, None, :], r.T) + centre[:, None, :]
        assert np.allclose(again, front, atol=1e-6)

    def test_side_views_are_opposite_quarter_turns(self):
        left = PoseCanonicalizer._VIEW_ROTATIONS[CanonicalView.LEFT_SIDE]
        right = PoseCanonicalizer._VIEW_ROTATIONS[CanonicalView.RIGHT_SIDE]
        assert np.allclose(left @ right, np.eye(3))

    def test_rotation_composes_as_documented(self, poses):
        # The returned rotation must equal view_rotation @ R_front, or a caller who
        # re-applies it by hand gets a different pose than the one they were handed.
        _, r_front = PoseCanonicalizer.transform_to_front_view(poses, return_rotation=True)
        for view, extra in PoseCanonicalizer._VIEW_ROTATIONS.items():
            _, r = PoseCanonicalizer.to_canonical_geometric(poses, view,
                                                            return_rotation=True)
            assert np.allclose(r, np.matmul(extra[None], r_front))


class TestDispatch:
    def test_string_views_match_enum_views(self, poses):
        for view in CanonicalView:
            assert np.array_equal(
                PoseCanonicalizer.to_canonical_geometric(poses, view.value),
                PoseCanonicalizer.to_canonical_geometric(poses, view),
            )

    def test_string_views_are_case_insensitive(self, poses):
        assert np.array_equal(
            PoseCanonicalizer.to_canonical_geometric(poses, "LEFT_SIDE"),
            PoseCanonicalizer.to_canonical_geometric(poses, CanonicalView.LEFT_SIDE),
        )

    def test_unknown_view_is_rejected(self, poses):
        with pytest.raises(ValueError):
            PoseCanonicalizer.to_canonical_geometric(poses, "sideways")

    def test_module_entry_point_matches_the_class(self, poses):
        for view in CanonicalView:
            assert np.array_equal(
                pt.canonicalize_pose(poses, view=view),
                PoseCanonicalizer.to_canonical_geometric(poses, view),
            )

    def test_geometric_is_the_default_model(self, poses):
        # None must mean "geometric", not "no canonicalization".
        assert np.array_equal(
            pt.canonicalize_pose(poses, model=None, view="front"),
            PoseCanonicalizer.to_canonical_geometric(poses, CanonicalView.FRONT),
        )

    def test_the_per_view_methods_are_gone(self):
        # They were four near-duplicate entry points for one derivation; `view=` replaced
        # them. If one comes back, so does the drift risk.
        for name in ("transform_to_back_view", "transform_to_left_side_view",
                     "transform_to_right_side_view"):
            assert not hasattr(PoseCanonicalizer, name), f"{name} reappeared"

    def test_determinism(self, poses):
        # The geometric path is closed-form: same input, same output, every time.
        a = PoseCanonicalizer.to_canonical_geometric(poses, CanonicalView.FRONT)
        b = PoseCanonicalizer.to_canonical_geometric(poses, CanonicalView.FRONT)
        assert np.array_equal(a, b)
