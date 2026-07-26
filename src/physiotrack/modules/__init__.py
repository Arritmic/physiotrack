"""Vendored model backends.

Each name below is resolved on first access rather than at import time. Every backend
carries a different slice of the deep-learning stack — Sapiens pulls torchvision, the
YOLO wrappers pull ultralytics, DDHPose pulls timm — so importing them all eagerly
meant that touching *any* backend loaded *all* of them. That made an unrelated import
such as ``from physiotrack.signals import joint_angles`` (pure NumPy/SciPy) reach the
3DPCNet canonicaliser and, through this module, drag in torch and ultralytics.

These are internal wiring, not public API; use the high-level predictors
([`Pose`][physiotrack.Pose], [`Detection`][physiotrack.Detection], ...) instead.
"""

# The lazy names below are invisible to anything that reads this file without running
# it -- type checkers, IDE completion, and the mkdocstrings/griffe pass that builds the
# API reference. Declaring them under TYPE_CHECKING makes the public surface statically
# resolvable while keeping the import cost at zero: the block never executes at runtime,
# so ``__getattr__`` still does the real work on first access.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DepthAnythingV2 import DepthAnythingV2Inference
    from .Sapiens import SapiensPoseEstimation, SapiensSegmentation, draw_segmentation_map
    from .SegFace import SegFaceInference
    from .ViTPose import VitInference
    from .Yolo import Detector, Segmentor, YoloPose
    from .ZipDepth import ZipDepthInference

_LAZY_ATTRS = {
    "SapiensPoseEstimation": ".Sapiens",
    "SapiensSegmentation": ".Sapiens",
    "draw_segmentation_map": ".Sapiens",
    "Detector": ".Yolo",
    "Segmentor": ".Yolo",
    "YoloPose": ".Yolo",
    "VitInference": ".ViTPose",
    "DepthAnythingV2Inference": ".DepthAnythingV2",
    "ZipDepthInference": ".ZipDepth",
    "SegFaceInference": ".SegFace",
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    """Import a backend on first access (PEP 562)."""
    if name in _LAZY_ATTRS:
        import importlib

        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache so later lookups skip this path
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
