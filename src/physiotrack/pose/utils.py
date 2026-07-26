from .config import COCO, COCO_WHOLEBODY, HALPE_TO_COCO_KEYPOINT_MAP, HUMAN26M
import json
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import warnings

from .._logging import get_logger

logger = get_logger(__name__)

class Keypoint:
    def __init__(self, id, x, y, confidence, keypoint_names):
        self.id = id
        self.x = x
        self.y = y
        self.confidence = confidence
        self._keypoint_names = keypoint_names
    
    @property
    def name(self):
        return self._keypoint_names.get(str(self.id), f"unknown_{self.id}")
    
class KeypointCollection:
    def __init__(self, keypoints_data, pose_archetecture):
        self._keypoint_names = COCO_WHOLEBODY if pose_archetecture == "WHOLEBODY" else COCO
        self._keypoints = {}
        self._keypoints_by_name = {}
        
        for kp_data in keypoints_data:
            kp = Keypoint(
                kp_data['id'], 
                kp_data['x'], 
                kp_data['y'], 
                kp_data['confidence'],
                self._keypoint_names
            )
            self._keypoints[kp.id] = kp
            self._keypoints_by_name[kp.name] = kp
    
    def id(self, keypoint_id):
        """Return keypoint by ID"""
        return self._keypoints.get(keypoint_id)
    
    def name(self, keypoint_name):
        """Return keypoint by name"""
        return self._keypoints_by_name.get(keypoint_name)

class PoseObject:
    def __init__(self, detection_data, pose_archetecture="WHOLEBODY"):
        self.id = detection_data['id']
        self.box = detection_data['box']  # [x1, y1, x2, y2]
        self.keypoints = KeypointCollection(detection_data['keypoints'], pose_archetecture)

class PoseObjectsFrame:
    def __init__(self, frame_data, pose_archetecture="WHOLEBODY"):
        self.frame_data = frame_data
        self.pose_archetecture = pose_archetecture
        self.pose_objects = []
        self.convert_frame_data_to_poses()

    def __iter__(self):
        """Make PoseObjects iterable"""
        return iter(self.pose_objects)
    
    def __len__(self):
        """Get number of poses"""
        return len(self.pose_objects)
    
    def __getitem__(self, index):
        """Get pose by index"""
        return self.pose_objects[index]
    
    def to_json(self):
        return self.frame_data

    def convert_frame_data_to_poses(self):
        """Convert frame_data to list of PoseObject instances"""
        for detection in self.frame_data['instances']:
            pose = PoseObject(detection, self.pose_archetecture)
            self.pose_objects.append(pose)
        return self.pose_objects


def add_3d_keypoints(frame_data, npy_data):
    """Attach lifted 3D keypoints to the subject they were computed from.

    The 3D lifting path is single-subject: the 2D sequence fed to the lifter is built
    from the first detection of each frame (see ``Pose3D._as_coco17_sequence``), so
    ``npy_data`` holds exactly one pose per frame. That pose is therefore attached only
    to the first detection.

    Earlier versions copied it onto *every* detection in the frame, which fabricated
    identical 3D poses for people who had never been lifted. When a frame contains more
    than one subject this emits a warning rather than inventing data for the others.

    Args:
        frame_data (list[dict]): Per-frame records, each with an ``"instances"`` list.
            Modified in place.
        npy_data (np.ndarray): Lifted poses of shape ``(N, 17, 3)`` in H36M joint order,
            one per frame.

    Returns:
        list[dict]: ``frame_data``, with ``"keypoints3D"`` set on the lifted subject.

    Warns:
        RuntimeWarning: If any frame holds multiple detections, since only one of them
            can receive a 3D pose.
    """
    multi_subject_frames = 0

    for frame_idx, frame in enumerate(frame_data):
        if frame_idx >= npy_data.shape[0]:
            continue
        detections = frame.get("instances") or []
        if not detections:
            continue
        if len(detections) > 1:
            multi_subject_frames += 1

        keypoints_3d = npy_data[frame_idx]  # (17, 3)
        # Index 0 is the subject the 2D sequence was extracted from.
        detections[0]["keypoints3D"] = [
            {
                "id": keypoint_idx,
                "x": float(keypoints_3d[keypoint_idx][0]),
                "y": float(keypoints_3d[keypoint_idx][1]),
                "z": float(keypoints_3d[keypoint_idx][2]),
                "name": HUMAN26M[keypoint_idx],
            }
            for keypoint_idx in range(17)
        ]

    if multi_subject_frames:
        warnings.warn(
            f"3D pose lifting is single-subject: {multi_subject_frames} frame(s) contain "
            f"more than one detection, and only the first received 'keypoints3D'. The "
            f"remaining subjects have no 3D pose rather than a copy of someone else's.",
            RuntimeWarning,
            stacklevel=2,
        )
    return frame_data


def coco17_to_halpe26(keypoints):
    """Convert a COCO-17 keypoint sequence to the Halpe-26 layout MotionBERT expects.

    COCO's 17 body joints are index-identical to Halpe's first 17, so only the three
    synthesised joints differ: ``17`` head (eye midpoint), ``18`` neck (centroid of
    nose and both shoulders) and ``19`` pelvis (hip midpoint). The remaining Halpe
    joints (toes and heels, ``20``-``25``) have no COCO equivalent and stay zero;
    MotionBERT's ``halpe2h36m`` does not read them.

    Args:
        keypoints (np.ndarray): ``(N, 17, 2)`` or ``(N, 17, 3)`` COCO keypoints. A
            missing confidence channel is filled with ``1.0``.

    Returns:
        np.ndarray: ``(N, 26, 3)`` float64 keypoints as ``(x, y, confidence)``.

    Raises:
        ValueError: If the input is not ``(N, 17, 2)`` or ``(N, 17, 3)``.
    """
    arr = np.asarray(keypoints, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] not in (2, 3):
        raise ValueError(
            f"Expected COCO-17 keypoints of shape (N, 17, 2) or (N, 17, 3), got "
            f"{tuple(arr.shape)}."
        )
    if arr.shape[2] == 2:  # no confidence channel: treat every joint as certain
        arr = np.concatenate([arr, np.ones((*arr.shape[:2], 1))], axis=2)

    out = np.zeros((arr.shape[0], 26, 3), dtype=np.float64)
    out[:, :17, :] = arr

    # Confidence of a synthesised joint is the weakest of its parents, so an unreliable
    # parent cannot silently produce a confident derived joint.
    out[:, 17, :2] = (arr[:, 1, :2] + arr[:, 2, :2]) / 2.0          # head <- eyes
    out[:, 17, 2] = np.minimum(arr[:, 1, 2], arr[:, 2, 2])
    out[:, 18, :2] = (arr[:, 0, :2] + arr[:, 5, :2] + arr[:, 6, :2]) / 3.0   # neck
    out[:, 18, 2] = np.minimum.reduce([arr[:, 0, 2], arr[:, 5, 2], arr[:, 6, 2]])
    out[:, 19, :2] = (arr[:, 11, :2] + arr[:, 12, :2]) / 2.0        # pelvis <- hips
    out[:, 19, 2] = np.minimum(arr[:, 11, 2], arr[:, 12, 2])
    return out


def coco17_to_h36m(keypoints):
    """Convert a COCO-17 keypoint sequence to the 17-joint Human3.6M layout.

    Human3.6M's root, spine, thorax and head joints are not observed by COCO, so they
    are derived: root and thorax from the hip and shoulder midpoints, spine from the
    midpoint of those two, and both neck/nose and head from COCO's nose.

    Args:
        keypoints (np.ndarray): ``(N, 17, 2)`` or ``(N, 17, 3)`` COCO keypoints. Any
            confidence channel is dropped -- the DDHPose backend takes ``(x, y)`` only.

    Returns:
        np.ndarray: ``(N, 17, 2)`` float64 keypoints in Human3.6M joint order.

    Raises:
        ValueError: If the input is not ``(N, 17, 2)`` or ``(N, 17, 3)``.
    """
    arr = np.asarray(keypoints, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] not in (2, 3):
        raise ValueError(
            f"Expected COCO-17 keypoints of shape (N, 17, 2) or (N, 17, 3), got "
            f"{tuple(arr.shape)}."
        )
    c = arr[..., :2]
    out = np.zeros((c.shape[0], 17, 2), dtype=np.float64)
    out[:, 1] = c[:, 12]    # right hip
    out[:, 2] = c[:, 14]    # right knee
    out[:, 3] = c[:, 16]    # right ankle
    out[:, 4] = c[:, 11]    # left hip
    out[:, 5] = c[:, 13]    # left knee
    out[:, 6] = c[:, 15]    # left ankle
    out[:, 11] = c[:, 5]    # left shoulder
    out[:, 12] = c[:, 7]    # left elbow
    out[:, 13] = c[:, 9]    # left wrist
    out[:, 14] = c[:, 6]    # right shoulder
    out[:, 15] = c[:, 8]    # right elbow
    out[:, 16] = c[:, 10]   # right wrist
    out[:, 0] = (c[:, 11] + c[:, 12]) / 2.0     # root <- hip midpoint
    out[:, 8] = (c[:, 5] + c[:, 6]) / 2.0       # thorax <- shoulder midpoint
    out[:, 7] = (out[:, 0] + out[:, 8]) / 2.0   # spine <- midpoint of those
    out[:, 9] = c[:, 0]     # neck/nose
    out[:, 10] = c[:, 0]    # head
    return out
