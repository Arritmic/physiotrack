"""2D/3D human pose estimation, keypoint naming, and pose evaluation metrics.

Only the keypoint name maps are loaded eagerly — they are plain dictionaries. The
predictors and the evaluation/canonicalisation helpers are resolved on first access,
because each carries part of the deep-learning stack. This matters beyond start-up
cost: the signals subsystem imports the name maps from here to label keypoints, and
would otherwise pull torch into what is a pure NumPy/SciPy analysis path.
"""

from .config import COCO_WHOLEBODY_NAMES, HUMAN26M_NAMES

# The lazy names below are invisible to anything that reads this file without running
# it -- type checkers, IDE completion, and the mkdocstrings/griffe pass that builds the
# API reference. Declaring them under TYPE_CHECKING makes the public surface statically
# resolvable while keeping the import cost at zero: the block never executes at runtime,
# so ``__getattr__`` still does the real work on first access.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canonicalizer import CanonicalView, PoseCanonicalizer, canonicalize_pose
    from .evaluate import (
        calculate_mpjpe,
        calculate_pampjpe,
        calculate_rotation_error,
        compare_canonicalization_methods,
        evaluate_canonicalization,
        evaluate_pose_predictions,
    )
    from .pose import Pose
    from .pose3D import Pose3D

_LAZY_ATTRS = {
    # predictors (pull their model backends)
    "Pose": ".pose",
    "Pose3D": ".pose3D",
    # canonicalisation (3DPCNet is a torch model)
    "PoseCanonicalizer": ".canonicalizer",
    "canonicalize_pose": ".canonicalizer",
    "CanonicalView": ".canonicalizer",
    # evaluation metrics (torch)
    "evaluate_pose_predictions": ".evaluate",
    "evaluate_canonicalization": ".evaluate",
    "calculate_mpjpe": ".evaluate",
    "calculate_pampjpe": ".evaluate",
    "calculate_rotation_error": ".evaluate",
    "compare_canonicalization_methods": ".evaluate",
}

__all__ = ["COCO_WHOLEBODY_NAMES", "HUMAN26M_NAMES"] + list(_LAZY_ATTRS)


def __getattr__(name):
    """Resolve the predictors and metric helpers on first access (PEP 562)."""
    if name in _LAZY_ATTRS:
        import importlib

        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache so later lookups skip this path
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
