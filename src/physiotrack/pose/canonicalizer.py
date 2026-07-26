import numpy as np
from typing import Tuple, Union, Optional, Dict, Any
import json
from enum import Enum
from pathlib import Path
import os
from .._logging import get_logger
from ..models import Models
from ..modules._3DCPNet.inference import canonicalize_poses_3dpcnet

logger = get_logger(__name__)
import warnings
from .._paths import weights_dir


class CanonicalView(Enum):
    """The canonical viewpoint a pose is rotated to.

    A canonicalization *parameter*, not a model: it selects which orientation
    [`canonicalize_pose`][physiotrack.canonicalize_pose] targets. Accepted anywhere a
    ``view=`` argument is taken, either as a member or as its string value.

    Attributes:
        FRONT: Subject facing the camera.
        BACK: Subject facing away from the camera.
        LEFT_SIDE: Subject in left-side profile.
        RIGHT_SIDE: Subject in right-side profile.
    """

    FRONT = "front"
    BACK = "back"
    LEFT_SIDE = "left_side"
    RIGHT_SIDE = "right_side"


CanonicalModels = Models.Pose3D.Canonicalizer.Models

class PoseCanonicalizer:
    """Transform 3D poses to viewpoint-invariant canonical orientations.

    Reorients Human3.6M-style 17-joint 3D poses into a canonical view (e.g. facing the
    camera) so that downstream analysis is invariant to the subject's global rotation.

    There is one transform entry point --
    [`to_canonical_geometric`][physiotrack.PoseCanonicalizer.to_canonical_geometric],
    taking the target ``view`` as an argument -- because all four views share a single
    derivation: the front view is computed from the torso plane, and the other three are
    that result rotated about the vertical axis. Two method families are available:

    - **Geometric**: a closed-form transform that fits a torso plane (shoulders +
      hips), aligns its normal with the camera axis and the shoulders with the
      X-axis, then optionally rotates to a back/left/right view. Deterministic and
      dependency-free. See
      [`to_canonical_geometric`][physiotrack.PoseCanonicalizer.to_canonical_geometric].
    - **3DPCNet**: a learned network that regresses the canonicalizing rotation.
      Weights are downloaded on first use. See
      [`canonicalize_3dpcnet`][physiotrack.PoseCanonicalizer.canonicalize_3dpcnet].

    Most users should call the module-level
    [`canonicalize_pose`][physiotrack.canonicalize_pose] helper, which dispatches to
    the right method based on the ``model`` argument, rather than these methods
    directly. Poses use the H36M joint ordering with the pelvis at index ``0``.

    Attributes:
        JOINT_INDICES (dict[str, int]): H36M joint indices used to build the torso
            plane: ``left_shoulder`` (11), ``right_shoulder`` (14), ``left_hip`` (4),
            ``right_hip`` (1).

    Example:
        ```python
        import numpy as np
        import physiotrack as pt

        poses = np.random.randn(100, 17, 3)  # (N frames, 17 joints, xyz)
        front = pt.PoseCanonicalizer.to_canonical_geometric(poses, view="front")
        ```

    See Also:
        [`canonicalize_pose`][physiotrack.canonicalize_pose]: recommended entry point.
    """
    
    # H36M joint indices
    JOINT_INDICES = {
        'left_shoulder': 11,
        'right_shoulder': 14,
        'left_hip': 4,
        'right_hip': 1
    }
    
    @staticmethod
    def extract_torso_plane(poses_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a torso plane from the shoulder and hip joints.

        Uses the four torso keypoints (left/right shoulders and hips) to define a
        plane: its center is their centroid and its normal is the cross product of
        the shoulder vector and the hip-to-shoulder torso vector. The normal is
        unit-normalized. This plane is the basis for the geometric canonicalization.

        Args:
            poses_3d (np.ndarray): Poses of shape ``(N, 17, 3)`` — ``N`` frames, 17
                H36M joints, ``(x, y, z)`` coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple ``(plane_center,
                normal_vector, torso_points)`` where ``plane_center`` is ``(N, 3)``,
                ``normal_vector`` is the unit torso-plane normal ``(N, 3)``, and
                ``torso_points`` are the stacked four torso joints ``(N, 4, 3)``.
        """
        left_shoulder = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['left_shoulder'], :]
        right_shoulder = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['right_shoulder'], :]
        left_hip = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['left_hip'], :]
        right_hip = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['right_hip'], :]
        
        # Stack for easier processing: (N, 4, 3)
        torso_points = np.stack([left_shoulder, right_shoulder, left_hip, right_hip], axis=1)
        
        # plane center (centroid of 4 points)
        plane_center = np.mean(torso_points, axis=1)  # (N, 3)
        
        shoulder_vector = right_shoulder - left_shoulder  # (N, 3)
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_vector = shoulder_center - hip_center  # (N, 3)
        # Normal to the torso plane
        normal_vector = np.cross(shoulder_vector, torso_vector)  # (N, 3)
        # Normalize
        normal_vector = normal_vector / (np.linalg.norm(normal_vector, axis=1, keepdims=True) + 1e-8)
        
        return plane_center, normal_vector, torso_points
    
    @staticmethod
    def compute_rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
        """Compute per-sample rotation matrices that align one vector onto another.

        Uses Rodrigues' rotation formula to build, for each sample, the rotation that
        maps ``from_vec`` onto ``to_vec``. Both inputs are normalized internally;
        near-parallel pairs fall back to the identity rotation.

        Args:
            from_vec (np.ndarray): Source vectors, shape ``(N, 3)``.
            to_vec (np.ndarray): Target vector(s), shape ``(N, 3)`` or a single
                ``(3,)`` vector broadcast to all ``N`` samples.

        Returns:
            np.ndarray: Rotation matrices of shape ``(N, 3, 3)`` such that rotating
                ``from_vec`` by them yields ``to_vec``.
        """
        # Ensure inputs are normalized
        from_vec = from_vec / (np.linalg.norm(from_vec, axis=-1, keepdims=True) + 1e-8)
        if to_vec.ndim == 1:
            to_vec = np.tile(to_vec, (from_vec.shape[0], 1))
        to_vec = to_vec / (np.linalg.norm(to_vec, axis=-1, keepdims=True) + 1e-8)
        # Cross product for rotation axis
        cross = np.cross(from_vec, to_vec)
        cross_norm = np.linalg.norm(cross, axis=-1, keepdims=True)
        # Dot product for rotation angle
        dot = np.sum(from_vec * to_vec, axis=-1, keepdims=True)
        # Handle parallel vectors
        parallel_mask = cross_norm.squeeze(-1) < 1e-6
        # Rodrigues' rotation formula
        N = from_vec.shape[0]
        R = np.eye(3)[None, :, :].repeat(N, axis=0)
        # For non-parallel vectors
        non_parallel = ~parallel_mask
        if np.any(non_parallel):
            k = cross[non_parallel] / (cross_norm[non_parallel] + 1e-8)
            theta = np.arccos(np.clip(dot[non_parallel], -1, 1))
            # Skew-symmetric matrix
            K = np.zeros((np.sum(non_parallel), 3, 3))
            K[:, 0, 1] = -k[:, 2]
            K[:, 0, 2] = k[:, 1]
            K[:, 1, 0] = k[:, 2]
            K[:, 1, 2] = -k[:, 0]
            K[:, 2, 0] = -k[:, 1]
            K[:, 2, 1] = k[:, 0]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R[non_parallel] = (np.eye(3)[None, :, :] + 
                              sin_theta[:, :, None] * K + 
                              (1 - cos_theta)[:, :, None] * np.matmul(K, K))
        return R
    
    @staticmethod
    def transform_to_front_view(poses_3d: np.ndarray, return_rotation: bool = False):
        """Rotate poses to the canonical front view.

        Two-step rotation: (1) align the torso-plane normal with the Z-axis so the
        torso plane becomes parallel to the XY plane (the subject faces the camera),
        then (2) rotate about Z by the minimal angle that makes the shoulders
        parallel to the X-axis. Poses are rotated about the torso-plane center and
        translated back, so the global position is preserved.

        Args:
            poses_3d (np.ndarray): Poses of shape ``(N, 17, 3)``.
            return_rotation (bool, optional): If ``True``, also return the combined
                rotation matrices. Defaults to ``False``.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: The front-view poses
                ``(N, 17, 3)``; if ``return_rotation`` is ``True``, a tuple of
                ``(poses, R_total)`` where ``R_total`` is ``(N, 3, 3)`` and equals
                ``R2 @ R1`` (the facing rotation followed by the shoulder-leveling
                rotation).
        """
        plane_center, normal_vector, _ = PoseCanonicalizer.extract_torso_plane(poses_3d)
        # Step 1: Align torso plane normal with Z-axis
        target_normal = np.array([0, 0, 1])
        R1 = PoseCanonicalizer.compute_rotation_matrix(normal_vector, target_normal)
        # Apply first rotation
        centered_poses = poses_3d - plane_center[:, None, :]
        rotated_once = np.matmul(centered_poses, R1.transpose(0, 2, 1))
        # Step 2: Align shoulders with X-axis using minimal rotation
        # Get shoulder positions after first rotation
        left_shoulder = rotated_once[:, PoseCanonicalizer.JOINT_INDICES['left_shoulder'], :]
        right_shoulder = rotated_once[:, PoseCanonicalizer.JOINT_INDICES['right_shoulder'], :]
        shoulder_vector = right_shoulder - left_shoulder
        # Project shoulder vector onto XY plane (remove Z component)
        shoulder_vector[:, 2] = 0
        shoulder_vector = shoulder_vector / (np.linalg.norm(shoulder_vector, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate current angle of shoulder vector in XY plane
        current_angle = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])  # (N,)
        # Determine minimal rotation to align with X-axis
        # Two options: rotate to 0° or to 180° (±π)
        angle_to_0 = -current_angle
        angle_to_pi = np.where(current_angle > 0, 
                            np.pi - current_angle,
                            -np.pi - current_angle)
        # Choose the rotation with smaller absolute angle
        use_angle_to_0 = np.abs(angle_to_0) <= np.abs(angle_to_pi)
        rotation_angle = np.where(use_angle_to_0, angle_to_0, angle_to_pi)
        # Create rotation matrices for Z-axis rotation
        N = poses_3d.shape[0]
        R2 = np.zeros((N, 3, 3))
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        R2[:, 0, 0] = cos_angle
        R2[:, 0, 1] = -sin_angle
        R2[:, 1, 0] = sin_angle
        R2[:, 1, 1] = cos_angle
        R2[:, 2, 2] = 1
        # Apply second rotation
        final_poses = np.matmul(rotated_once, R2.transpose(0, 2, 1))
        # Translate back (optional)
        final_poses = final_poses + plane_center[:, None, :]
        
        if return_rotation:
            """
            R1: Get the person facing forward (torso plane parallel to camera)
            R2: Straighten the person (shoulders horizontal)
            """
            # Combine both rotations: R_total = R2 @ R1
            R_total = np.matmul(R2, R1)
            return final_poses, R_total
        return final_poses
    
    # Every non-front canonical view is the front view plus one fixed rotation about
    # the vertical axis. Naming them here keeps a single derivation for all four.
    _VIEW_ROTATIONS = {
        CanonicalView.BACK: np.array([[-1, 0, 0],       # 180 deg
                                      [0, 1, 0],
                                      [0, 0, -1]], dtype=float),
        CanonicalView.LEFT_SIDE: np.array([[0, 0, -1],  # 90 deg counter-clockwise
                                           [0, 1, 0],
                                           [1, 0, 0]], dtype=float),
        CanonicalView.RIGHT_SIDE: np.array([[0, 0, 1],  # 90 deg clockwise
                                            [0, 1, 0],
                                            [-1, 0, 0]], dtype=float),
    }

    @staticmethod
    def _rotate_from_front(poses_3d: np.ndarray, rotation: np.ndarray,
                           return_rotation: bool = False):
        """Canonicalize to the front view, then apply one further rotation.

        Args:
            poses_3d (np.ndarray): Input poses of shape ``(N, 17, 3)``.
            rotation (np.ndarray): A ``(3, 3)`` rotation applied about the torso-plane
                centre after the front-view transform.
            return_rotation (bool, optional): Also return the combined per-frame
                rotations. Defaults to ``False``.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: The rotated poses
                ``(N, 17, 3)``, or ``(poses, R_total)`` where
                ``R_total = rotation @ R_front`` with shape ``(N, 3, 3)``.
        """
        if return_rotation:
            front, r_front = PoseCanonicalizer.transform_to_front_view(
                poses_3d, return_rotation=True)
        else:
            front = PoseCanonicalizer.transform_to_front_view(poses_3d)

        # Rotate about the torso-plane centre so the subject spins in place rather than
        # orbiting the origin.
        centre, _, _ = PoseCanonicalizer.extract_torso_plane(front)
        rotated = np.matmul(front - centre[:, None, :], rotation.T) + centre[:, None, :]

        if return_rotation:
            batch = np.tile(rotation[None, :, :], (poses_3d.shape[0], 1, 1))
            return rotated, np.matmul(batch, r_front)
        return rotated

    @staticmethod
    def canonicalize_3dpcnet(poses_3d: np.ndarray, 
                            model: Union[CanonicalModels, None] = None,
                            view: Union[CanonicalView, str] = "front",
                            checkpoint_path: Optional[str] = None, 
                            config_path: Optional[str] = None,
                            device: str = 'cuda',
                            apply_transform: bool = True,
                            verbose: bool = False,
                            return_rotation: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Canonicalize poses with a learned 3DPCNet model.

        Resolves and (on first use) downloads the requested 3DPCNet checkpoint, then
        runs network inference to regress the canonicalizing rotation and canonical
        pose. Only the front view is supported; other views trigger a warning and
        fall back to front. Prefer the module-level
        [`canonicalize_pose`][physiotrack.canonicalize_pose] wrapper.

        Args:
            poses_3d (np.ndarray): Input poses of shape ``(N, 17, 3)``.
            model (Models.Pose3D.Canonicalizer.Models, optional): Which pretrained
                checkpoint to use — one of ``_3DPCNetS2``, ``_3DPCNetS3``,
                ``_3DPCNetTC48_byCam``, ``_3DPCNetTC48_byAction``. ``GEOMETRIC`` and
                ``None`` skip download and rely on ``checkpoint_path``. Defaults to
                ``None``.
            view (CanonicalView | str, optional): Target view.
                Only ``"front"`` is supported; others warn and use front. Accepts the
                enum or its string value. Defaults to ``"front"``.
            checkpoint_path (str, optional): Explicit checkpoint path that overrides
                the ``model`` enum lookup/download. Defaults to ``None``.
            config_path (str, optional): Explicit model-config YAML path. Defaults to
                ``None`` (uses the bundled ``inference_config.yaml``).
            device (str, optional): Inference device, ``"cuda"`` or ``"cpu"``.
                Defaults to ``"cuda"`` (falls back to CPU if CUDA is unavailable).
            apply_transform (bool, optional): If ``True``, apply the standard-to-3DPCNet
                coordinate transform to the input; if ``False``, assume the input is
                already in 3DPCNet format. Defaults to ``True``.
            verbose (bool, optional): If ``True``, print progress messages. Defaults
                to ``False``.
            return_rotation (bool, optional): If ``True``, also return the predicted
                rotation matrices. Defaults to ``False``.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: Canonicalized poses
                ``(N, 17, 3)``; if ``return_rotation`` is ``True``, a tuple of
                ``(poses, rotation_matrices)`` with rotations of shape ``(N, 3, 3)``.

        Note:
            The first call for a given model downloads its weights via
            [`Models.download_model`][physiotrack.Models].

        See Also:
            [`to_canonical_geometric`][physiotrack.PoseCanonicalizer.to_canonical_geometric]:
                the closed-form alternative.
        """
        
        if isinstance(view, CanonicalView):
            view = view.value
        
        if view != 'front':
            warnings.warn(
                "3DPCNet supports only the front view; applying the front view instead of "
                f"{view!r}.", RuntimeWarning, stacklevel=2)
        
        # Handle model download if model enum is provided
        if model is not None and model != CanonicalModels.GEOMETRIC:
            if checkpoint_path is None:
                # Download model if not exists
                checkpoint_file = os.path.join(str(weights_dir()), model.value)
                
                if not os.path.isfile(checkpoint_file):
                    logger.info('Downloading canonicalizer weights: %s', model.name)
                    Models.download_model(model)
                
                checkpoint_path = checkpoint_file
            
        # All 3DPCNet checkpoints share one architecture, so they load from a single
        # bundled inference config (no per-model YAML download required).
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'modules', '_3DCPNet',
                'configs', 'inference_config.yaml'
            )
        
        # Use the inference module function
        result = canonicalize_poses_3dpcnet(
            poses_3d,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            apply_transform=apply_transform,
            verbose=verbose,
            return_rotation=return_rotation
        )
        
        if verbose:
            # Per call: debug level, since this runs once per frame in a video loop.
            logger.debug('Canonicalized pose using the 3DPCNet method')
        return result
    
    @staticmethod
    def to_canonical_geometric(poses_3d: np.ndarray, view: Union[CanonicalView, str],
                                    return_rotation: bool = False,
                                    verbose: bool = False):
        """Canonicalize poses to a given view using the closed-form geometric method.

        Dispatches to the front/back/left/right transform based on ``view``. The
        result is deterministic and requires no model weights.

        Args:
            poses_3d (np.ndarray): Input poses of shape ``(N, 17, 3)``.
            view (CanonicalView | str): Target view — one of
                ``"front"``, ``"back"``, ``"left_side"``, ``"right_side"`` (or the
                corresponding ``View`` enum). Strings are lower-cased.
            return_rotation (bool, optional): If ``True``, also return the rotation
                matrices. Defaults to ``False``.
            verbose (bool, optional): If ``True``, print a progress message. Defaults
                to ``False``.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: Canonicalized poses
                ``(N, 17, 3)``; if ``return_rotation`` is ``True``, a tuple of
                ``(poses, R_total)`` with rotations of shape ``(N, 3, 3)``.

        Raises:
            ValueError: If ``view`` is not one of the four supported views.

        See Also:
            [`canonicalize_3dpcnet`][physiotrack.PoseCanonicalizer.canonicalize_3dpcnet]:
                the learned alternative.
        """
        if isinstance(view, str):
            view = CanonicalView(view.lower())
        if verbose:
            logger.debug('Canonicalized pose to the %s view using the geometric method',
                     view.value)
        
        if view is CanonicalView.FRONT:
            return PoseCanonicalizer.transform_to_front_view(poses_3d, return_rotation)
        if view not in PoseCanonicalizer._VIEW_ROTATIONS:
            raise ValueError(f"Unsupported view: {view}. Choose from {list(CanonicalView)}")
        return PoseCanonicalizer._rotate_from_front(
            poses_3d, PoseCanonicalizer._VIEW_ROTATIONS[view], return_rotation)
    
    @staticmethod
    def process_json_file(json_path: str, output_path: Optional[str] = None, 
                         view: Union[CanonicalView, str] = "front",
                         model: Union[CanonicalModels, None] = CanonicalModels.GEOMETRIC) -> Dict[str, Any]:
        """Canonicalize 3D poses stored in a detection-results JSON file.

        Reads a list of per-frame detection records, extracts the ``keypoints_3d``,
        canonicalizes them via [`canonicalize_pose`][physiotrack.canonicalize_pose],
        writes the canonicalized keypoints back into each record (plus
        ``canonical_view_applied`` and ``canonical_model`` fields), and optionally
        saves the updated JSON.

        Args:
            json_path (str): Path to the input JSON file with 3D keypoints.
            output_path (str, optional): Where to save the processed JSON. Defaults
                to ``None`` (no file written; data returned only).
            view (CanonicalView | str, optional): Target canonical
                view. Defaults to ``"front"``.
            model (Models.Pose3D.Canonicalizer.Models, optional): Canonicalization
                model — ``GEOMETRIC``, ``_3DPCNetS2``, ``_3DPCNetS3``, etc. Defaults
                to ``Models.Pose3D.Canonicalizer.Models.GEOMETRIC``.

        Returns:
            dict[str, Any]: The (possibly updated) detection data with canonicalized
                3D poses.

        See Also:
            [`process_npy_file`][physiotrack.PoseCanonicalizer.process_npy_file]:
                the ``.npy`` array equivalent.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract 3D keypoints from detection data
        poses_3d = PoseCanonicalizer._extract_3d_from_json(data)
        
        if poses_3d is not None:
            # Apply canonical transformation
            canonical_poses = canonicalize_pose(poses_3d, model, view)
            
            # Update the data with canonical poses
            data = PoseCanonicalizer._update_json_with_3d(data, canonical_poses, view, model)
            
            # Save if output path provided
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
        
        return data
    
    @staticmethod
    def process_npy_file(npy_path: str, output_path: Optional[str] = None,
                        view: Union[CanonicalView, str] = "front",
                        model: Union[CanonicalModels, None] = CanonicalModels.GEOMETRIC) -> np.ndarray:
        """Canonicalize 3D poses stored in a NumPy ``.npy`` file.

        Loads a ``(N, 17, 3)`` pose array, canonicalizes it via
        [`canonicalize_pose`][physiotrack.canonicalize_pose], and optionally saves
        the result to disk.

        Args:
            npy_path (str): Path to the input ``.npy`` file with 3D poses.
            output_path (str, optional): Where to save the canonicalized array.
                Defaults to ``None`` (no file written; array returned only).
            view (CanonicalView | str, optional): Target canonical
                view. Defaults to ``"front"``.
            model (Models.Pose3D.Canonicalizer.Models, optional): Canonicalization
                model. Defaults to ``Models.Pose3D.Canonicalizer.Models.GEOMETRIC``.

        Returns:
            np.ndarray: The canonicalized poses, shape ``(N, 17, 3)``.

        Example:
            ```python
            import physiotrack as pt

            canonical = pt.PoseCanonicalizer.process_npy_file(
                "output/X3D.npy",
                output_path="output/X3D_canonical.npy",
                view=pt.CanonicalView.FRONT,
                model=pt.Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
            )
            ```

        See Also:
            [`process_json_file`][physiotrack.PoseCanonicalizer.process_json_file]:
                the JSON equivalent.
        """
        poses_3d = np.load(npy_path)
        canonical_poses = canonicalize_pose(poses_3d, model, view)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, canonical_poses)
        
        return canonical_poses
    
    @staticmethod
    def _extract_3d_from_json(data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract 3D keypoints from detection JSON data.
        
        Args:
            data: Detection data dictionary
            
        Returns:
            np.ndarray | None: 3D poses array or None if not found
        """
        # Check if data contains 3D keypoints
        if not isinstance(data, list) or len(data) == 0:
            return None
        
        # Collect all 3D keypoints
        poses_3d_list = []
        for frame_data in data:
            if 'keypoints_3d' in frame_data:
                poses_3d_list.append(frame_data['keypoints_3d'])
        
        if len(poses_3d_list) > 0:
            poses_3d = np.array(poses_3d_list)
            # Reshape if necessary (assuming 17 joints with 3 coordinates each)
            if poses_3d.ndim == 2:
                poses_3d = poses_3d.reshape(-1, 17, 3)
            return poses_3d
        
        return None
    
    @staticmethod
    def _update_json_with_3d(data: Dict[str, Any], poses_3d: np.ndarray, 
                            view: Union[CanonicalView, str], model: Union[CanonicalModels, None]) -> Dict[str, Any]:
        """
        Update detection JSON data with canonicalized 3D poses.
        
        Args:
            data: Original detection data
            poses_3d: Canonicalized 3D poses
            view: Applied canonical view
            model: Applied canonicalization model
            
        Returns:
            dict[str, Any]: Updated detection data
        """
        if isinstance(view, str):
            view = CanonicalView(view.lower())
            
        if isinstance(data, list) and len(data) == len(poses_3d):
            for i, frame_data in enumerate(data):
                if 'keypoints_3d' in frame_data:
                    # Update with canonicalized poses
                    frame_data['keypoints_3d'] = poses_3d[i].flatten().tolist()
                    frame_data['canonical_view_applied'] = view.value
                    frame_data['canonical_model'] = model.name if model else 'GEOMETRIC'
        
        return data


def canonicalize_pose(poses_3d: np.ndarray, 
                     model: Union[CanonicalModels, None] = CanonicalModels.GEOMETRIC,
                     view: Union[CanonicalView, str] = "front",
                     return_rotation: bool = False,
                     apply_transform: bool = True,
                     verbose: bool = True):
    """Canonicalize 3D poses to a viewpoint-invariant orientation.

    Primary entry point for pose canonicalization. Dispatches to the closed-form
    geometric method when ``model`` is ``GEOMETRIC`` (or ``None``), or to a learned
    3DPCNet model for the ``_3DPCNet*`` enum values. Poses use the H36M 17-joint
    layout with the pelvis at index ``0``.

    Args:
        poses_3d (np.ndarray): Input poses of shape ``(N, 17, 3)``.
        model (Models.Pose3D.Canonicalizer.Models, optional): Canonicalization model.
            ``GEOMETRIC`` (or ``None``) uses the geometric transform;
            ``_3DPCNetS2``, ``_3DPCNetS3``, ``_3DPCNetTC48_byCam``,
            ``_3DPCNetTC48_byAction`` use the learned network. Defaults to
            ``Models.Pose3D.Canonicalizer.Models.GEOMETRIC``.
        view (CanonicalView | str, optional): Target view — one of
            ``"front"``, ``"back"``, ``"left_side"``, ``"right_side"``. 3DPCNet models
            only support ``"front"``. Defaults to ``"front"``.
        return_rotation (bool, optional): If ``True``, also return the rotation
            matrices ``(N, 3, 3)``. Supported by both methods. Defaults to ``False``.
        apply_transform (bool, optional): 3DPCNet only — if ``True`` apply the
            standard-to-3DPCNet coordinate transform to the input; if ``False`` the
            input is assumed already in 3DPCNet format. Defaults to ``True``.
        verbose (bool, optional): If ``True``, print progress messages. Defaults to
            ``True``.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: Canonicalized poses
            ``(N, 17, 3)``; if ``return_rotation`` is ``True``, a tuple of
            ``(poses, rotation_matrices)`` with rotations of shape ``(N, 3, 3)``.

    Raises:
        ValueError: If ``model`` is not a supported canonicalization model, or (via
            the geometric path) if ``view`` is not a supported view.

    Example:
        ```python
        import numpy as np
        import physiotrack as pt

        poses = np.random.randn(100, 17, 3)

        # Geometric front-view canonicalization
        canonical = pt.canonicalize_pose(
            poses,
            model=pt.Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
            view=pt.CanonicalView.FRONT,
        )

        # Learned 3DPCNet (auto-downloads weights on first use)
        dpcnet = pt.canonicalize_pose(
            poses, model=pt.Models.Pose3D.Canonicalizer.Models._3DPCNetS2, view="front",
        )
        ```

    See Also:
        [`PoseCanonicalizer`][physiotrack.PoseCanonicalizer]: the underlying methods.
        [`evaluate_canonicalization`][physiotrack.evaluate_canonicalization]: scoring
            canonicalization quality.
    """
    if model is None or model == CanonicalModels.GEOMETRIC:
        return PoseCanonicalizer.to_canonical_geometric(poses_3d, view, return_rotation)
    elif model in [CanonicalModels._3DPCNetS2, CanonicalModels._3DPCNetS3,
                   CanonicalModels._3DPCNetTC48_byCam, CanonicalModels._3DPCNetTC48_byAction]:
        result = PoseCanonicalizer.canonicalize_3dpcnet(poses_3d, model, view, 
                                                        apply_transform=apply_transform, 
                                                        verbose=verbose,
                                                        return_rotation=return_rotation)
        return result
    else:
        raise ValueError(f"Unsupported canonicalization model: {model}. Choose from {list(CanonicalModels)}")
