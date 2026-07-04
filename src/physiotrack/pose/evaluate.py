"""
Evaluation metrics for 3D pose estimation and canonicalization.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union


def compute_similarity_transform(X: np.ndarray, Y: np.ndarray, 
                                compute_optimal_scale: bool = False) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Computes the similarity transform (rotation, translation, scale) that best aligns Y to X.
    
    Args:
        X: Target poses, shape (N, 3) where N is number of points
        Y: Input poses to be transformed, shape (N, 3)
        compute_optimal_scale: If True, compute optimal scale; if False, scale = 1
        
    Returns:
        d: Squared error after transformation
        Z: Transformed Y
        T: Rotation matrix (3, 3)
        b: Scale factor
        c: Translation vector (3,)
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # Centered Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # Scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # Optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def calculate_mpjpe(preds: np.ndarray, gts: np.ndarray) -> float:
    """Compute the Mean Per Joint Position Error (MPJPE).

    MPJPE is the mean, over all joints and all samples, of the Euclidean distance
    between each predicted joint and its ground-truth position:
    ``mean_{n,j} || preds[n, j] - gts[n, j] ||_2``. No alignment is applied, so the
    predictions and ground truth must already share the same coordinate frame,
    scale, and root placement. For a rigid-alignment-invariant variant use
    [`calculate_pampjpe`][physiotrack.calculate_pampjpe].

    Args:
        preds (np.ndarray): Predicted poses, shape ``(N, J, 3)`` where ``N`` is the
            number of samples and ``J`` the number of joints. Coordinates are in
            arbitrary length units (e.g. meters).
        gts (np.ndarray): Ground-truth poses, shape ``(N, J, 3)``, in the same
            units and frame as ``preds``.

    Returns:
        float: The MPJPE, in the same units as the inputs (e.g. meters).

    Raises:
        AssertionError: If ``preds`` and ``gts`` do not have identical shapes.

    Example:
        ```python
        import numpy as np
        import physiotrack as pt

        preds = np.random.randn(64, 17, 3)
        gts = np.random.randn(64, 17, 3)
        err = pt.calculate_mpjpe(preds, gts)  # meters; multiply by 1000 for mm
        ```

    See Also:
        [`calculate_pampjpe`][physiotrack.calculate_pampjpe]: Procrustes-aligned MPJPE.
        [`evaluate_pose_predictions`][physiotrack.evaluate_pose_predictions]: bundles
            MPJPE with other metrics and rescales to millimeters.
    """
    assert preds.shape == gts.shape, f"Shape mismatch: preds {preds.shape} vs gts {gts.shape}"
    
    # Compute Euclidean distance for each joint
    distances = np.linalg.norm(preds - gts, axis=-1)  # (N, J)
    
    # Mean over all joints and samples
    mpjpe = np.mean(distances)
    
    return mpjpe


def calculate_pampjpe(preds: np.ndarray, gts: np.ndarray) -> float:
    """Compute the Procrustes-Aligned Mean Per Joint Position Error (PA-MPJPE).

    For each sample, the predicted pose is first optimally aligned to the ground
    truth with a similarity transform (rotation, translation, and scale) solved via
    Procrustes analysis, then MPJPE is computed on the aligned pose. Because it
    removes rigid misalignment and global scale, PA-MPJPE (also called "reconstruction
    error") isolates the quality of the pose *shape* independent of orientation and
    size. The per-sample errors are averaged over the batch.

    Args:
        preds (np.ndarray): Predicted poses, shape ``(N, J, 3)`` where ``N`` is the
            number of samples and ``J`` the number of joints.
        gts (np.ndarray): Ground-truth poses, shape ``(N, J, 3)``, in the same
            units as ``preds``.

    Returns:
        float: The PA-MPJPE, in the same units as the inputs (e.g. meters).

    Raises:
        AssertionError: If ``preds`` and ``gts`` do not have identical shapes.

    Example:
        ```python
        import physiotrack as pt

        pa = pt.calculate_pampjpe(preds, gts)  # rigid + scale aligned error
        ```

    See Also:
        [`calculate_mpjpe`][physiotrack.calculate_mpjpe]: unaligned position error.
    """
    assert preds.shape == gts.shape, f"Shape mismatch: preds {preds.shape} vs gts {gts.shape}"
    
    N = preds.shape[0]
    num_joints = preds.shape[1]
    
    pampjpe_per_sample = []
    
    for n in range(N):
        frame_pred = preds[n]  # (J, 3)
        frame_gt = gts[n]      # (J, 3)
        
        # Apply similarity transform to align prediction with ground truth
        _, _, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred_aligned = (b * frame_pred.dot(T)) + c
        
        # Compute error after alignment
        joint_errors = np.linalg.norm(frame_pred_aligned - frame_gt, axis=-1)  # (J,)
        pampjpe_per_sample.append(np.mean(joint_errors))
    
    pampjpe = np.mean(pampjpe_per_sample)
    
    return pampjpe


def calculate_rotation_error(pred_rotation: np.ndarray, gt_rotation: np.ndarray, 
                            return_degrees: bool = True,
                            method: str = 'frobenius') -> float:
    """Compute the mean rotation error between predicted and ground-truth rotations.

    Two error definitions are available. ``"frobenius"`` (the default, matching the
    3DPCNet reference implementation) averages the Frobenius norm of the matrix
    difference ``||R_pred - R_gt||_F`` and treats that value as radians. ``"geodesic"``
    computes the true relative rotation angle per sample,
    ``angle = arccos((trace(R_gt @ R_pred^T) - 1) / 2)``, and averages it. The
    Frobenius variant is not a true angle but is retained for parity with published
    3DPCNet numbers; prefer ``"geodesic"`` for a physically meaningful angular error.

    Args:
        pred_rotation (np.ndarray): Predicted rotation matrices, shape ``(N, 3, 3)``.
        gt_rotation (np.ndarray): Ground-truth rotation matrices, shape ``(N, 3, 3)``.
        return_degrees (bool, optional): If ``True`` return the error in degrees;
            if ``False`` return it in radians. Defaults to ``True``.
        method (str, optional): Error definition, one of ``"frobenius"`` (default,
            3DPCNet-compatible) or ``"geodesic"`` (true relative rotation angle).
            Defaults to ``"frobenius"``.

    Returns:
        float: The mean rotation error, in degrees if ``return_degrees`` else radians.

    Raises:
        AssertionError: If ``pred_rotation`` and ``gt_rotation`` do not have
            identical shapes.

    Example:
        ```python
        import physiotrack as pt

        err_deg = pt.calculate_rotation_error(pred_R, gt_R, method="geodesic")
        ```

    See Also:
        [`evaluate_canonicalization`][physiotrack.evaluate_canonicalization]: reports
            this rotation error alongside pose metrics.
    """
    assert pred_rotation.shape == gt_rotation.shape, f"Shape mismatch: {pred_rotation.shape} vs {gt_rotation.shape}"
    
    if method == 'frobenius':
        # Match 3DPCNet's approach: Frobenius norm treated as radians
        rotation_error = np.mean(np.linalg.norm(pred_rotation - gt_rotation, axis=(1, 2)))
        if return_degrees:
            rotation_error = rotation_error * 180.0 / np.pi
    else:  # geodesic
        N = pred_rotation.shape[0]
        errors = []
        
        for i in range(N):
            # Compute relative rotation: R_rel = R_gt @ R_pred^T
            R_rel = np.dot(gt_rotation[i], pred_rotation[i].T)
            
            # Extract angle from rotation matrix using trace
            # angle = arccos((trace(R) - 1) / 2)
            trace = np.trace(R_rel)
            trace = np.clip(trace, -1.0, 3.0)  # Numerical stability
            angle = np.arccos((trace - 1.0) / 2.0)
            
            errors.append(angle)
        
        rotation_error = np.mean(errors)
        
        if return_degrees:
            rotation_error = np.degrees(rotation_error)
    
    return rotation_error


def evaluate_pose_predictions(preds: np.ndarray, gts: np.ndarray, 
                             scale: float = 1000.0) -> Dict[str, float]:
    """Evaluate 3D pose predictions with a bundle of position-error metrics.

    Computes MPJPE and PA-MPJPE together with per-joint error statistics, and
    multiplies every reported value by ``scale`` (so passing ``scale=1000`` reports
    results in millimeters when the inputs are in meters). Torch tensors are accepted
    and are detached and moved to CPU automatically.

    Args:
        preds (np.ndarray | torch.Tensor): Predicted poses, shape ``(N, J, 3)``.
        gts (np.ndarray | torch.Tensor): Ground-truth poses, shape ``(N, J, 3)``.
        scale (float, optional): Multiplier applied to every returned metric, e.g.
            ``1000.0`` to convert meters to millimeters. Defaults to ``1000.0``.

    Returns:
        dict[str, float]: A dictionary with keys:

            - ``"mpjpe"`` (float): Mean Per Joint Position Error, scaled.
            - ``"pampjpe"`` (float): Procrustes-aligned MPJPE, scaled.
            - ``"mpjpe_per_joint"`` (list[float]): Per-joint mean error, length ``J``, scaled.
            - ``"max_joint_error"`` (float): Largest per-joint mean error, scaled.
            - ``"min_joint_error"`` (float): Smallest per-joint mean error, scaled.

    Example:
        ```python
        import physiotrack as pt

        metrics = pt.evaluate_pose_predictions(preds, gts, scale=1000.0)
        print(f"MPJPE: {metrics['mpjpe']:.2f} mm")
        ```

    See Also:
        [`evaluate_canonicalization`][physiotrack.evaluate_canonicalization]: adds
            canonicalization-specific pose and rotation errors.
        [`calculate_mpjpe`][physiotrack.calculate_mpjpe],
        [`calculate_pampjpe`][physiotrack.calculate_pampjpe]: the underlying metrics.
    """
    # Convert to numpy if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.detach().cpu().numpy()
    
    # Calculate metrics
    mpjpe = calculate_mpjpe(preds, gts) * scale
    pampjpe = calculate_pampjpe(preds, gts) * scale
    
    # Additional metrics
    # Per-joint errors
    joint_errors = np.mean(np.linalg.norm(preds - gts, axis=-1), axis=0) * scale  # (J,)
    
    return {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'mpjpe_per_joint': joint_errors.tolist(),
        'max_joint_error': np.max(joint_errors),
        'min_joint_error': np.min(joint_errors)
    }


def evaluate_canonicalization(pred_canonical: np.ndarray, gt_canonical: np.ndarray,
                             pred_rotation: Optional[np.ndarray] = None, 
                             gt_rotation: Optional[np.ndarray] = None,
                             scale: float = 1000.0) -> Dict[str, float]:
    """Evaluate pose canonicalization results, including optional rotation error.

    Runs [`evaluate_pose_predictions`][physiotrack.evaluate_pose_predictions] on the
    predicted vs. ground-truth canonical poses, adds a direct L2 pose error, and — when
    both rotation arguments are supplied — the rotation error in degrees and radians.
    Torch tensors are accepted for any input and are converted to NumPy automatically.

    Args:
        pred_canonical (np.ndarray | torch.Tensor): Predicted canonical poses, shape
            ``(N, J, 3)``.
        gt_canonical (np.ndarray | torch.Tensor): Ground-truth canonical poses, shape
            ``(N, J, 3)``.
        pred_rotation (np.ndarray | torch.Tensor, optional): Predicted rotation
            matrices, shape ``(N, 3, 3)``. Defaults to ``None`` (rotation metrics
            skipped).
        gt_rotation (np.ndarray | torch.Tensor, optional): Ground-truth rotation
            matrices, shape ``(N, 3, 3)``. Defaults to ``None`` (rotation metrics
            skipped).
        scale (float, optional): Multiplier applied to pose errors, e.g. ``1000.0``
            to report millimeters. Defaults to ``1000.0``.

    Returns:
        dict[str, float]: All keys from
            [`evaluate_pose_predictions`][physiotrack.evaluate_pose_predictions]
            (``"mpjpe"``, ``"pampjpe"``, ``"mpjpe_per_joint"``, ``"max_joint_error"``,
            ``"min_joint_error"``) plus:

            - ``"pose_error_mm"`` (float): Mean per-joint L2 distance between
              predicted and ground-truth canonical poses, scaled by ``scale``.
            - ``"rotation_error_deg"`` (float): Mean rotation error in degrees.
              Present only when both rotation arguments are given.
            - ``"rotation_error_rad"`` (float): Mean rotation error in radians.
              Present only when both rotation arguments are given.

    Example:
        ```python
        import physiotrack as pt

        canonical, rotation = pt.canonicalize_pose(
            poses, model=pt.Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
            view="front", return_rotation=True,
        )
        metrics = pt.evaluate_canonicalization(
            canonical, gt_canonical,
            pred_rotation=rotation, gt_rotation=gt_rotation, scale=1000.0,
        )
        print(metrics["mpjpe"], metrics.get("rotation_error_deg"))
        ```

    See Also:
        [`canonicalize_pose`][physiotrack.canonicalize_pose]: produces the canonical
            poses and rotations evaluated here.
        [`calculate_rotation_error`][physiotrack.calculate_rotation_error]: the
            rotation metric used internally.
    """
    # Convert to numpy if needed
    if isinstance(pred_canonical, torch.Tensor):
        pred_canonical = pred_canonical.detach().cpu().numpy()
    if isinstance(gt_canonical, torch.Tensor):
        gt_canonical = gt_canonical.detach().cpu().numpy()
    
    # Pose metrics
    metrics = evaluate_pose_predictions(pred_canonical, gt_canonical, scale=scale)
    
    # Add pose error (direct L2 distance)
    pose_error = np.mean(np.linalg.norm(pred_canonical - gt_canonical, axis=-1)) * scale
    metrics['pose_error_mm'] = pose_error
    
    # Rotation metrics if provided
    if pred_rotation is not None and gt_rotation is not None:
        if isinstance(pred_rotation, torch.Tensor):
            pred_rotation = pred_rotation.detach().cpu().numpy()
        if isinstance(gt_rotation, torch.Tensor):
            gt_rotation = gt_rotation.detach().cpu().numpy()
        
        rotation_error_deg = calculate_rotation_error(pred_rotation, gt_rotation, return_degrees=True)
        rotation_error_rad = calculate_rotation_error(pred_rotation, gt_rotation, return_degrees=False)
        
        metrics.update({
            'rotation_error_deg': rotation_error_deg,
            'rotation_error_rad': rotation_error_rad
        })
    
    return metrics


def compare_canonicalization_methods(poses_3d: np.ndarray, 
                                    gt_canonical: Optional[np.ndarray] = None,
                                    gt_rotation: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare different canonicalization methods on the same input poses.
    
    Args:
        poses_3d: Input poses, shape (N, J, 3)
        gt_canonical: Optional ground truth canonical poses
        gt_rotation: Optional ground truth rotation matrices
        
    Returns:
        Dictionary with metrics for each method
    """
    from .canonicalizer import canonicalize_pose
    from ..models import Models
    
    results = {}
    
    # Geometric method
    try:
        geometric_canonical = canonicalize_pose(
            poses_3d,
            model=Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
            view=Models.Pose3D.Canonicalizer.View.FRONT
        )
        
        if gt_canonical is not None:
            results['geometric'] = evaluate_pose_predictions(
                geometric_canonical, gt_canonical, scale=1000.0
            )
        else:
            results['geometric'] = {'status': 'computed', 'shape': geometric_canonical.shape}
    except Exception as e:
        results['geometric'] = {'error': str(e)}
    
    # 3DPCNet S2 method
    try:
        s2_canonical = canonicalize_pose(
            poses_3d,
            model=Models.Pose3D.Canonicalizer.Models._3DPCNetS2,
            view=Models.Pose3D.Canonicalizer.View.FRONT
        )
        
        if gt_canonical is not None:
            results['3dpcnet_s2'] = evaluate_pose_predictions(
                s2_canonical, gt_canonical, scale=1000.0
            )
        else:
            results['3dpcnet_s2'] = {'status': 'computed', 'shape': s2_canonical.shape}
    except Exception as e:
        results['3dpcnet_s2'] = {'error': str(e)}
    
    # 3DPCNet S3 method (if available)
    try:
        s3_canonical = canonicalize_pose(
            poses_3d,
            model=Models.Pose3D.Canonicalizer.Models._3DPCNetS3,
            view=Models.Pose3D.Canonicalizer.View.FRONT
        )
        
        if gt_canonical is not None:
            results['3dpcnet_s3'] = evaluate_pose_predictions(
                s3_canonical, gt_canonical, scale=1000.0
            )
        else:
            results['3dpcnet_s3'] = {'status': 'computed', 'shape': s3_canonical.shape}
    except Exception as e:
        results['3dpcnet_s3'] = {'error': str(e)}
    
    return results