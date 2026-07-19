"""Physiotrack — a Python toolkit for contactless human understanding.

Quick start::

    import physiotrack as pt

    det = pt.Detection.Person()
    result = det.predict(frame)          # -> pt.Result
    annotated = result.plot()

Every image predictor (Detection, Pose, Segmentation, Depth, Face) exposes
``.predict()`` (and is callable), returning a unified result object whose
``.plot()`` draws the overlay. See ``docs/API_REDESIGN.md`` for the full API.
"""

# --- Low-level backends -------------------------------------------------------
# These are imported here because the high-level subpackages import them FROM the
# top-level package (internal wiring). They are not part of the documented public
# API (see __all__), but remain importable for advanced use.
from .modules import SapiensPoseEstimation
from .modules import SapiensSegmentation, draw_segmentation_map
from .modules import Detector
from .modules import Segmentor
from .modules import YoloPose
from .modules import VitInference
from .modules import DepthAnythingV2Inference
from .modules import ZipDepthInference

# --- Model registry -----------------------------------------------------------
from .models import Models

# --- Trackers -----------------------------------------------------------------
from .trackers import BYTETracker, StrongSORT, OCSort, BoostTrack, Tracker, TrackerConfig

# --- High-level predictors ----------------------------------------------------
from .detect import Detection
from .segment import Segmentation
from .pose import Pose
from .depth import Depth
from .face import Face, VRFace, FaceOrientation

# --- Orchestrator -------------------------------------------------------------
from .capture.video import Video

# --- Unified result objects ---------------------------------------------------
from .results import Result, DepthResult, TrackResult, Instance, Keypoint, Keypoints

# --- Pose 3D backends (kept importable for advanced use) ----------------------
from .modules.MotionBERT.inference import MotionBERTInference
from .modules.DDHPose.inference import DDHPoseInference

# --- Pose post-processing -----------------------------------------------------
from .pose.canonicalizer import PoseCanonicalizer, canonicalize_pose
from .modules._3DCPNet.inference import apply_3dpcnet_transform, reverse_3dpcnet_transform
from .pose.evaluate import (
    evaluate_pose_predictions,
    evaluate_canonicalization,
    calculate_mpjpe,
    calculate_pampjpe,
    calculate_rotation_error,
)


# ``Pose3D`` is imported lazily: its module pulls in heavy/optional 3D-rendering
# dependencies (e.g. smplx) that should not be required just to ``import physiotrack``.
def __getattr__(name):
    if name == "Pose3D":
        from .pose.pose3D import Pose3D
        return Pose3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # predictors
    "Detection", "Pose", "Pose3D", "Segmentation", "Depth",
    "Face", "VRFace", "FaceOrientation",
    # tracking
    "Tracker", "TrackerConfig",
    # orchestrator
    "Video",
    # registry
    "Models",
    # results
    "Result", "DepthResult", "TrackResult", "Instance", "Keypoint", "Keypoints",
    # pose post-processing
    "PoseCanonicalizer", "canonicalize_pose",
    "apply_3dpcnet_transform", "reverse_3dpcnet_transform",
    "evaluate_pose_predictions", "evaluate_canonicalization",
    "calculate_mpjpe", "calculate_pampjpe", "calculate_rotation_error",
]
