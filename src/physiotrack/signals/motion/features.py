import numpy as np
import pandas as pd


# Single source of truth for the anatomical joint-angle definitions, shared by
# ``compute_all_joint_angles`` and the ``JointAnglePlotter`` overlay so the two
# can never drift. Each entry maps a joint name to a COCO-17 keypoint triplet
# (A, vertex B, C); the angle is measured at the middle joint B.
JOINT_ANGLE_TRIPLETS = {
    "leftShoulder": (7, 5, 11),   # left_elbow, left_shoulder, left_hip
    "rightShoulder": (8, 6, 12),  # right_elbow, right_shoulder, right_hip
    "leftElbow": (5, 7, 9),       # left_shoulder, left_elbow, left_wrist
    "rightElbow": (6, 8, 10),     # right_shoulder, right_elbow, right_wrist
    "leftHip": (5, 11, 13),       # left_shoulder, left_hip, left_knee
    "rightHip": (6, 12, 14),      # right_shoulder, right_hip, right_knee
    "leftKnee": (11, 13, 15),     # left_hip, left_knee, left_ankle
    "rightKnee": (12, 14, 16),    # right_hip, right_knee, right_ankle
}


# Clinical range-of-motion (ROM) / goniometry definitions. Unlike the interior
# joint angles above, a ROM movement measures a distal limb segment against a body
# reference axis, named by the anatomical movement and zeroed to a neutral pose.
# Each entry: vertex (joint), ref (defines the reference axis vertex->ref),
# moving (distal point, vertex->moving), and ``value = scale * raw + offset`` where
# ``raw`` is the angle (deg) between the two vectors. COCO-17 ids:
#   shoulders 5/6, hips 11/12, knees 13/14.
ROM_DEFINITIONS = {
    # Sagittal plane (thigh vs trunk); flexion and extension share the geometry.
    "leftHipFlexion":    dict(vertex=11, ref=5,  moving=13, scale=-1.0, offset=180.0, label="L Hip Flex"),
    "rightHipFlexion":   dict(vertex=12, ref=6,  moving=14, scale=-1.0, offset=180.0, label="R Hip Flex"),
    "leftHipExtension":  dict(vertex=11, ref=5,  moving=13, scale=-1.0, offset=180.0, label="L Hip Ext"),
    "rightHipExtension": dict(vertex=12, ref=6,  moving=14, scale=-1.0, offset=180.0, label="R Hip Ext"),
    # Frontal plane (thigh vs pelvis line); abduction is zeroed, adduction is raw.
    "leftHipAbduction":  dict(vertex=11, ref=12, moving=13, scale=1.0,  offset=-90.0, label="L Hip Abd"),
    "rightHipAbduction": dict(vertex=12, ref=11, moving=14, scale=1.0,  offset=-90.0, label="R Hip Abd"),
    "leftHipAdduction":  dict(vertex=11, ref=12, moving=13, scale=1.0,  offset=0.0,   label="L Hip Add"),
    "rightHipAdduction": dict(vertex=12, ref=11, moving=14, scale=1.0,  offset=0.0,   label="R Hip Add"),
}

# Two distinct planes per hip, useful as a default overlay (avoids drawing the
# duplicate flexion/extension and abduction/adduction arcs, which share geometry).
DEFAULT_ROM_MOVEMENTS = [
    "leftHipFlexion", "rightHipFlexion",
    "leftHipAbduction", "rightHipAbduction",
]

# Distinct, clear colors (BGR), one per ROM movement, shared by the angle-panel
# rows and the skeleton arcs so each case is the same color in both places.
_ROM_PALETTE = [
    (235, 150, 30),   # blue
    (200, 190, 30),   # cyan
    (50, 170, 60),    # green
    (40, 200, 150),   # lime
    (20, 140, 255),   # orange
    (40, 60, 230),    # red
    (200, 60, 175),   # purple
    (230, 60, 230),   # magenta
]


def rom_color(movement_name):
    """Distinct BGR color for a ROM movement (stable per movement name)."""
    keys = list(ROM_DEFINITIONS.keys())
    if movement_name in keys:
        return _ROM_PALETTE[keys.index(movement_name) % len(_ROM_PALETTE)]
    return (60, 60, 235)  # fallback


def compute_rom_angles(keypoints, movements=None, conf_threshold=0.3):
    """Compute clinical range-of-motion (ROM) angles (degrees) from 2D keypoints.

    ROM differs from an interior joint angle: it measures a distal limb segment
    against a body reference axis and reports it as a clinically named movement
    (hip flexion / extension / abduction / adduction), zeroed to a neutral pose.
    Each movement is defined in ``ROM_DEFINITIONS`` by a vertex joint, a reference
    point (giving the reference axis ``vertex -> ref``), a moving distal point
    (``vertex -> moving``) and an affine map ``value = scale * raw + offset``, where
    ``raw`` is the angle between the two vectors. Only the confident movements are
    returned (all three involved keypoints must clear ``conf_threshold``).

    Args:
        keypoints (list[dict]): One frame's keypoints as ``{"id", "x", "y",
            "confidence"}`` dicts, COCO-17 (shoulders ``5``/``6``, hips ``11``/``12``,
            knees ``13``/``14``).
        movements (list[str], optional): Movement names from ``ROM_DEFINITIONS`` (e.g.
            ``"leftHipFlexion"``, ``"rightHipAbduction"``). Defaults to ``None`` (all
            movements in ``ROM_DEFINITIONS``).
        conf_threshold (float, optional): Minimum confidence in ``[0, 1]`` required for
            each of the three keypoints of a movement. Defaults to ``0.3``.

    Returns:
        dict[str, float]: Movement name -> angle in degrees, for the confidently
            measured movements only. Empty if ``keypoints`` is empty/falsy.

    Example:
        ```python
        import physiotrack as pt
        from physiotrack.signals import compute_rom_angles

        result = pt.Pose.Person().predict(frame)
        kps = result.to_dict()["detections"][0]["keypoints"]
        rom = compute_rom_angles(kps, movements=["leftHipFlexion", "rightHipFlexion"])
        ```

    See Also:
        [`joint_angles`][physiotrack.signals.joint_angles]: interior anatomical angles.
        [`JointAnglePlotter`][physiotrack.signals.JointAnglePlotter]: renders ROM as an overlay.
    """
    if not keypoints:
        return {}
    if movements is None:
        movements = list(ROM_DEFINITIONS.keys())
    kp = {k["id"]: k for k in keypoints}
    out = {}
    for name in movements:
        spec = ROM_DEFINITIONS[name]
        v, r, m = kp.get(spec["vertex"]), kp.get(spec["ref"]), kp.get(spec["moving"])
        if not (v and r and m):
            continue
        if min(v["confidence"], r["confidence"], m["confidence"]) < conf_threshold:
            continue
        rad = compute_joint_angle_2d((r["x"], r["y"]), (v["x"], v["y"]), (m["x"], m["y"]))
        if rad is None or np.isnan(rad):
            continue
        out[name] = spec["scale"] * float(np.degrees(rad)) + spec["offset"]
    return out


def joint_angles(keypoints, joints=None, conf_threshold=0.3):
    """Interior anatomical joint angles (degrees) from one frame's keypoints.

    Measures the interior angle at each joint vertex from a COCO-17 keypoint triplet
    defined in ``JOINT_ANGLE_TRIPLETS`` (angle A-B-C measured at the middle joint B),
    via the cosine rule. Use this directly on pose-estimated keypoints -- no plotter
    required. The eight supported joints are ``leftShoulder``, ``rightShoulder``,
    ``leftElbow``, ``rightElbow``, ``leftHip``, ``rightHip``, ``leftKnee``, ``rightKnee``.

    Args:
        keypoints (list[dict]): One frame's keypoints as ``{"id", "x", "y",
            "confidence"}`` dicts (COCO-17).
        joints (list[str], optional): Subset of ``JOINT_ANGLE_TRIPLETS`` keys. Defaults
            to ``None`` (all eight joints).
        conf_threshold (float, optional): Minimum confidence in ``[0, 1]`` required for
            each of the three keypoints of an angle. Defaults to ``0.3``.

    Returns:
        dict[str, float]: ``{joint_name: degrees}`` for the confidently measured joints
            only. Empty if ``keypoints`` is empty/falsy.

    Example:
        ```python
        import physiotrack as pt
        from physiotrack.signals import joint_angles

        result = pt.Pose.Person().predict(frame)
        kps = result.to_dict()["detections"][0]["keypoints"]
        angles = joint_angles(kps, joints=["leftElbow", "rightElbow"])
        ```

    See Also:
        [`compute_rom_angles`][physiotrack.signals.compute_rom_angles]: clinical ROM movements.
        [`compute_all_joint_angles`][physiotrack.signals.compute_all_joint_angles]:
            angles over a whole DataFrame sequence.
    """
    if not keypoints:
        return {}
    if joints is None:
        joints = list(JOINT_ANGLE_TRIPLETS.keys())
    kp = {k["id"]: k for k in keypoints}
    out = {}
    for joint in joints:
        a_id, b_id, c_id = JOINT_ANGLE_TRIPLETS[joint]
        a, b, c = kp.get(a_id), kp.get(b_id), kp.get(c_id)
        if not (a and b and c):
            continue
        if min(a["confidence"], b["confidence"], c["confidence"]) < conf_threshold:
            continue
        rad = compute_joint_angle_2d((a["x"], a["y"]), (b["x"], b["y"]), (c["x"], c["y"]))
        if rad is not None and not np.isnan(rad):
            out[joint] = float(np.degrees(rad))
    return out


def compute_velocity(rel_df):
    """
    Computes the velocity for each keypoint coordinate (2D and 3D) in the relative DataFrame.
    Velocity is computed as: V_t = (P_t - P_{t-1}) / dt
    where dt is dynamically computed using the 'time' column.
    Args:
        rel_df (pd.DataFrame): DataFrame with relative keypoint coordinates.
    Returns:
        pd.DataFrame: A new DataFrame with velocity columns.
    """
    velocity_df = pd.DataFrame(index=rel_df.index)
    
    if "time" not in rel_df.columns:
        raise ValueError("The 'time' column is required to compute velocity but is missing.")
    
    dt = rel_df["time"].diff()  # Compute per-frame time difference
    
    # Find 2D keypoint columns
    keypoint_2d_columns = [col for col in rel_df.columns 
                          if (col.endswith('_x') or col.endswith('_y')) 
                          and not col.startswith('3d_')
                          and col != 'time']
    
    # Find 3D keypoint columns
    keypoint_3d_columns = [col for col in rel_df.columns 
                          if col.startswith('3d_') 
                          and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))
                          and col != 'time']
    
    # Compute 2D velocities using V_t = (P_t - P_{t-1}) / dt
    for col in keypoint_2d_columns:
        velocity_df[f"vel_2d_{col}"] = rel_df[col].diff() / dt
    
    # Compute 3D velocities using V_t = (P_t - P_{t-1}) / dt
    for col in keypoint_3d_columns:
        # Remove '3d_' prefix from column name for cleaner naming
        clean_col = col.replace('3d_', '')
        velocity_df[f"vel_3d_{clean_col}"] = rel_df[col].diff() / dt
    
    return velocity_df


def compute_acceleration(rel_df):
    """
    Computes the acceleration for each keypoint coordinate (2D and 3D) in the relative DataFrame.
    Acceleration is computed as: A_t = (P_t - 2 * P_{t-1} + P_{t-2}) / (dt^2)
    where dt is dynamically computed using the 'time' column.
    
    Args:
        rel_df (pd.DataFrame): DataFrame with relative keypoint coordinates.
    Returns:
        pd.DataFrame: A new DataFrame with acceleration columns.
    """
    acceleration_df = pd.DataFrame(index=rel_df.index)
    
    if "time" not in rel_df.columns:
        raise ValueError("The 'time' column is required to compute acceleration but is missing.")
    
    dt = rel_df["time"].diff()  # Compute per-frame time difference
    dt_squared = dt ** 2        # Compute dt^2
    
    # Find 2D keypoint columns
    keypoint_2d_columns = [col for col in rel_df.columns 
                          if (col.endswith('_x') or col.endswith('_y')) 
                          and not col.startswith('3d_')
                          and col != 'time']
    
    # Find 3D keypoint columns
    keypoint_3d_columns = [col for col in rel_df.columns 
                          if col.startswith('3d_') 
                          and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))
                          and col != 'time']
    
    # Compute 2D accelerations using A_t = (P_t - 2 * P_{t-1} + P_{t-2}) / (dt^2)
    for col in keypoint_2d_columns:
        # Manual calculation: P_t - 2*P_{t-1} + P_{t-2}
        P_t = rel_df[col]
        P_t_minus_1 = rel_df[col].shift(1)
        P_t_minus_2 = rel_df[col].shift(2)
        acceleration_df[f"acc_2d_{col}"] = (P_t - 2 * P_t_minus_1 + P_t_minus_2) / dt_squared
    
    # Compute 3D accelerations using A_t = (P_t - 2 * P_{t-1} + P_{t-2}) / (dt^2)
    for col in keypoint_3d_columns:
        # Manual calculation: P_t - 2*P_{t-1} + P_{t-2}
        P_t = rel_df[col]
        P_t_minus_1 = rel_df[col].shift(1)
        P_t_minus_2 = rel_df[col].shift(2)
        # Remove '3d_' prefix from column name for cleaner naming
        clean_col = col.replace('3d_', '')
        acceleration_df[f"acc_3d_{clean_col}"] = (P_t - 2 * P_t_minus_1 + P_t_minus_2) / dt_squared
    
    return acceleration_df


def compute_joint_angle_2d(A, B, C):
    """
    Computes the joint angle at point B given three 2D keypoints A, B, and C using the cosine rule.
    Args:
        A, B, C: Array-like or tuple with two elements (x, y) representing the coordinates.
    Returns:
        float: The angle in radians at point B.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    # Avoid division by zero
    if norm_BA == 0 or norm_BC == 0:
        return np.nan
    
    cosine_angle = dot_product / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1, 1)  # Clip for numerical stability
    angle = np.arccos(cosine_angle)
    return angle


def compute_joint_angle_3d(A, B, C):
    """
    Computes the joint angle at point B given three 3D keypoints A, B, and C using the cosine rule.
    Args:
        A, B, C: Array-like or tuple with three elements (x, y, z) representing the coordinates.
    Returns:
        float: The angle in radians at point B.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    # Avoid division by zero
    if norm_BA == 0 or norm_BC == 0:
        return np.nan
    
    cosine_angle = dot_product / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1, 1)  # Clip for numerical stability
    angle = np.arccos(cosine_angle)
    return angle


def compute_all_joint_angles(rel_df):
    """Compute per-frame interior joint angles over a whole keypoint DataFrame.

    Applies the cosine-rule interior-angle measurement (see
    [`joint_angles`][physiotrack.signals.joint_angles]) row by row to a wide keypoint
    DataFrame, for every joint in ``JOINT_ANGLE_TRIPLETS``. 2D and 3D angles are
    computed independently, each only if the corresponding coordinate columns exist.
    Angles are returned in **radians** (from ``compute_joint_angle_2d/3d``); rows with
    missing/NaN coordinates yield ``NaN``.

    The eight COCO-17 triplets (A, vertex B, C) are:

    - ``leftShoulder`` (7, 5, 11), ``rightShoulder`` (8, 6, 12)
    - ``leftElbow`` (5, 7, 9), ``rightElbow`` (6, 8, 10)
    - ``leftHip`` (5, 11, 13), ``rightHip`` (6, 12, 14)
    - ``leftKnee`` (11, 13, 15), ``rightKnee`` (12, 14, 16)

    Args:
        rel_df (pandas.DataFrame): Wide keypoint DataFrame (e.g. from
            [`extract_keypoints_sequence`][physiotrack.signals.extract_keypoints_sequence]
            or [`get_relative_coordinates`][physiotrack.signals.get_relative_coordinates]),
            with columns ``"{k}_x"``/``"{k}_y"`` (2D) and/or ``"3d_{k}_x"``/``"3d_{k}_y"``/
            ``"3d_{k}_z"`` (3D).

    Returns:
        pandas.DataFrame: New DataFrame (same index) with one column per joint present:
            ``"ang_2d_{joint}"`` and/or ``"ang_3d_{joint}"``, values in radians.

    See Also:
        [`compute_all_motion_features`][physiotrack.signals.compute_all_motion_features]:
            velocity + acceleration + these angles concatenated onto ``rel_df``.
    """
    joint_triplets = JOINT_ANGLE_TRIPLETS

    df_angles = pd.DataFrame(index=rel_df.index)
    
    has_2d_data = any(col.endswith('_x') and not col.startswith('3d_') for col in rel_df.columns)
    has_3d_data = any(col.startswith('3d_') and col.endswith('_x') for col in rel_df.columns)
    
    for joint, triplet in joint_triplets.items():
        A_joint, B_joint, C_joint = triplet
        
        # Compute 2D joint angles if 2D data exists
        if has_2d_data:
            col_name_2d = f"ang_2d_{joint}"
            
            def compute_angle_row_2d(row):
                try:
                    A_coord = (row[f"{A_joint}_x"], row[f"{A_joint}_y"])
                    B_coord = (row[f"{B_joint}_x"], row[f"{B_joint}_y"])
                    C_coord = (row[f"{C_joint}_x"], row[f"{C_joint}_y"])
                except KeyError:
                    return np.nan
                
                # Check for NaN values
                if any(pd.isna(x) for x in (A_coord + B_coord + C_coord)):
                    return np.nan
                
                return compute_joint_angle_2d(A_coord, B_coord, C_coord)
            
            df_angles[col_name_2d] = rel_df.apply(compute_angle_row_2d, axis=1)
        
        # Compute 3D joint angles if 3D data exists
        if has_3d_data:
            col_name_3d = f"ang_3d_{joint}"
            
            def compute_angle_row_3d(row):
                try:
                    A_coord = (row[f"3d_{A_joint}_x"], row[f"3d_{A_joint}_y"], row[f"3d_{A_joint}_z"])
                    B_coord = (row[f"3d_{B_joint}_x"], row[f"3d_{B_joint}_y"], row[f"3d_{B_joint}_z"])
                    C_coord = (row[f"3d_{C_joint}_x"], row[f"3d_{C_joint}_y"], row[f"3d_{C_joint}_z"])
                except KeyError:
                    return np.nan
                
                # Check for NaN values
                if any(pd.isna(x) for x in (A_coord + B_coord + C_coord)):
                    return np.nan
                
                return compute_joint_angle_3d(A_coord, B_coord, C_coord)
            
            df_angles[col_name_3d] = rel_df.apply(compute_angle_row_3d, axis=1)
    
    return df_angles


def compute_all_motion_features(rel_df):
    """Compute velocity, acceleration and joint angles and concatenate onto the input.

    One-call feature builder: runs ``compute_velocity`` (finite-difference first
    derivative), ``compute_acceleration`` (second derivative) and
    [`compute_all_joint_angles`][physiotrack.signals.compute_all_joint_angles], then
    column-concatenates all three onto ``rel_df``. The velocity/acceleration helpers
    require a ``"time"`` column (used to compute ``dt``).

    Args:
        rel_df (pandas.DataFrame): Wide keypoint DataFrame with a ``"time"`` column,
            typically the pelvis-relative coordinates from
            [`get_relative_coordinates`][physiotrack.signals.get_relative_coordinates].

    Returns:
        pandas.DataFrame: ``rel_df`` plus columns ``vel_2d_*``/``vel_3d_*``,
            ``acc_2d_*``/``acc_3d_*`` and ``ang_2d_*``/``ang_3d_*``.

    Raises:
        ValueError: If ``rel_df`` has no ``"time"`` column (needed for the derivatives).

    Example:
        ```python
        from physiotrack.signals import (
            extract_keypoints_sequence, get_relative_coordinates, compute_all_motion_features,
        )

        kp_df = extract_keypoints_sequence(data, candidate_key_points=list(range(17)) + [135])
        rel = get_relative_coordinates(kp_df, reference_point_id=135)
        motion_df = compute_all_motion_features(rel)
        ```
    """
    # velocity
    velocity_df = compute_velocity(rel_df)
    # acceleration
    acceleration_df = compute_acceleration(rel_df)
    # joint angles
    angles_df = compute_all_joint_angles(rel_df)
    motion_features_df = pd.concat([rel_df, velocity_df, acceleration_df, angles_df], axis=1)
    return motion_features_df


def get_relative_coordinates(df, reference_point_id=135):
    """Recenter keypoint coordinates relative to a chosen reference keypoint.

    Subtracts the reference keypoint's position from every other keypoint, per frame,
    for both 2D (``_x``/``_y``) and 3D (``3d_*_x``/``_y``/``_z``) columns that are
    present. This makes the motion translation-invariant (e.g. pelvis-centered), which
    is the usual pre-step before computing velocity/acceleration/angles. The reference
    point's own coordinate columns are dropped; non-coordinate columns (and ``frame``/
    ``time``) are copied through unchanged.

    Args:
        df (pandas.DataFrame): Wide keypoint DataFrame from
            [`extract_keypoints_sequence`][physiotrack.signals.extract_keypoints_sequence].
        reference_point_id (int, optional): Keypoint id to use as the origin. Defaults
            to ``135`` (the synthesized pelvis from
            [`add_pelvic_centroid`][physiotrack.signals.add_pelvic_centroid]). Its 2D
            and/or 3D columns must exist in ``df``.

    Returns:
        pandas.DataFrame: New DataFrame of recentered coordinates plus passed-through
            non-coordinate columns (e.g. ``time``, ``frame``, ``detection_id``).

    Raises:
        ValueError: If neither 2D nor 3D columns for ``reference_point_id`` are found.

    See Also:
        [`compute_all_motion_features`][physiotrack.signals.compute_all_motion_features]:
            the typical next step.
    """
    # Define reference point column names for 2D and 3D
    ref_x_2d = f"{reference_point_id}_x"
    ref_y_2d = f"{reference_point_id}_y"
    ref_x_3d = f"3d_{reference_point_id}_x"
    ref_y_3d = f"3d_{reference_point_id}_y"
    ref_z_3d = f"3d_{reference_point_id}_z"
    
    # Check if reference point exists in the data
    has_2d_ref = ref_x_2d in df.columns and ref_y_2d in df.columns
    has_3d_ref = ref_x_3d in df.columns and ref_y_3d in df.columns and ref_z_3d in df.columns
    
    if not has_2d_ref and not has_3d_ref:
        raise ValueError(f"Reference point ID {reference_point_id} not found in the data.")
    
    relative_data = pd.DataFrame()
    
    # Get reference coordinates
    if has_2d_ref:
        center_x_2d = df[ref_x_2d]
        center_y_2d = df[ref_y_2d]
    
    if has_3d_ref:
        center_x_3d = df[ref_x_3d]
        center_y_3d = df[ref_y_3d]
        center_z_3d = df[ref_z_3d]
    
    # Identify all keypoint columns (2D and 3D) excluding reference point
    reference_cols = [ref_x_2d, ref_y_2d, ref_x_3d, ref_y_3d, ref_z_3d]
    
    # Process 2D keypoints
    if has_2d_ref:
        keypoint_2d_cols = [col for col in df.columns 
                           if (col.endswith('_x') or col.endswith('_y')) 
                           and not col.startswith('3d_')
                           and col not in reference_cols]
        
        for col in keypoint_2d_cols:
            if col.endswith('_x'):
                relative_data[col] = df[col] - center_x_2d
            elif col.endswith('_y'):
                relative_data[col] = df[col] - center_y_2d
    
    # Process 3D keypoints
    if has_3d_ref:
        keypoint_3d_cols = [col for col in df.columns 
                           if col.startswith('3d_') 
                           and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))
                           and col not in reference_cols]
        
        for col in keypoint_3d_cols:
            if col.endswith('_x'):
                relative_data[col] = df[col] - center_x_3d
            elif col.endswith('_y'):
                relative_data[col] = df[col] - center_y_3d
            elif col.endswith('_z'):
                relative_data[col] = df[col] - center_z_3d
    
    # Copy non-coordinate columns (like frame, time, etc.)
    non_coord_cols = [col for col in df.columns 
                      if not (col.endswith('_x') or col.endswith('_y') or col.endswith('_z') or col.endswith('_confidence'))
                      or col in ['frame', 'time']]
    
    for col in non_coord_cols:
        if col in df.columns:
            relative_data[col] = df[col]
    
    return relative_data


def get_keypoint_features(motion_df, keypoint_id_2d, keypoint_id_3d=None):
    """Slice out one keypoint's feature columns as separate 2D and 3D DataFrames.

    Selects, from a full motion-feature DataFrame, the coordinate / velocity /
    acceleration columns for a single keypoint plus all joint-angle columns
    (``ang_2d_*`` in the 2D frame, ``ang_3d_*`` in the 3D frame), always prefixed by
    the base columns ``["time", "frame", "detection_id"]``. Only columns that actually
    exist are kept.

    Args:
        motion_df (pandas.DataFrame): Output of
            [`compute_all_motion_features`][physiotrack.signals.compute_all_motion_features].
        keypoint_id_2d (int): 2D keypoint id to extract (COCO-WholeBody).
        keypoint_id_3d (int, optional): 3D keypoint id (Human3.6M). Defaults to ``None``,
            meaning use the same id as ``keypoint_id_2d``.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: ``(features_2d, features_3d)``. Each
            is empty if none of its columns are present.

    Example:
        ```python
        from physiotrack.pose import COCO_WHOLEBODY_NAMES, HUMAN26M_NAMES
        from physiotrack.signals import get_keypoint_features

        f2d, f3d = get_keypoint_features(
            motion_df,
            int(COCO_WHOLEBODY_NAMES["left_wrist"]),
            int(HUMAN26M_NAMES["left_wrist"]),
        )
        ```

    See Also:
        [`select_feature_data`][physiotrack.signals.select_feature_data]: pick which
            feature columns (coordinates / velocity / ...) to work with.
    """
    if keypoint_id_3d is None:
        keypoint_id_3d = keypoint_id_2d
    
    base_cols = ['time', 'frame', 'detection_id']
    
    # 2D features
    cols_2d = base_cols.copy()
    for col in motion_df.columns:
        # 2D coordinates, velocity, acceleration
        if col in [f'{keypoint_id_2d}_x', f'{keypoint_id_2d}_y'] or \
           col in [f'vel_2d_{keypoint_id_2d}_x', f'vel_2d_{keypoint_id_2d}_y'] or \
           col in [f'acc_2d_{keypoint_id_2d}_x', f'acc_2d_{keypoint_id_2d}_y'] or \
           col.startswith('ang_2d_'):
            cols_2d.append(col)
    
    # 3D features
    cols_3d = base_cols.copy()
    for col in motion_df.columns:
        # 3D coordinates, velocity, acceleration
        if col in [f'3d_{keypoint_id_3d}_x', f'3d_{keypoint_id_3d}_y', f'3d_{keypoint_id_3d}_z'] or \
           col in [f'vel_3d_{keypoint_id_3d}_x', f'vel_3d_{keypoint_id_3d}_y', f'vel_3d_{keypoint_id_3d}_z'] or \
           col in [f'acc_3d_{keypoint_id_3d}_x', f'acc_3d_{keypoint_id_3d}_y', f'acc_3d_{keypoint_id_3d}_z'] or \
           col.startswith('ang_3d_'):
            cols_3d.append(col)
    
    existing_cols_2d = [col for col in cols_2d if col in motion_df.columns]
    existing_cols_3d = [col for col in cols_3d if col in motion_df.columns]
    
    features_2d = motion_df[existing_cols_2d] if existing_cols_2d else pd.DataFrame()
    features_3d = motion_df[existing_cols_3d] if existing_cols_3d else pd.DataFrame()
    
    return features_2d, features_3d


def select_feature_data(keypoint_id_2d, keypoint_id_3d, feature_type='coordinates'):
    """Build the 2D/3D column names for a given feature type of one keypoint.

    A naming helper: returns the exact motion-feature column names to read for the
    requested ``feature_type``, for both the 2D and 3D keypoint. Handy for pulling the
    right series out of the DataFrame returned by
    [`get_keypoint_features`][physiotrack.signals.get_keypoint_features].

    Args:
        keypoint_id_2d (int): 2D keypoint id (COCO-WholeBody).
        keypoint_id_3d (int): 3D keypoint id (Human3.6M).
        feature_type (str, optional): One of ``"coordinates"``, ``"velocity"``,
            ``"acceleration"`` or ``"angles"``. Defaults to ``"coordinates"``. For
            ``"angles"`` the returned names are the fixed elbow-angle columns
            (``ang_2d_leftElbow``/``ang_2d_rightElbow`` and the 3D equivalents), not
            keypoint-specific.

    Returns:
        dict[str, list[str]]: ``{"2d_features": [...], "3d_features": [...]}`` column
            names. Returns ``None`` if ``feature_type`` is not recognized.

    Example:
        ```python
        from physiotrack.signals import select_feature_data

        sel = select_feature_data(9, 9, feature_type="velocity")
        vx = keypoint_df_2d[sel["2d_features"][0]]
        ```
    """

    if feature_type == 'coordinates':
        return {
            '2d_features': [f'{keypoint_id_2d}_x', f'{keypoint_id_2d}_y'],
            '3d_features': [f'3d_{keypoint_id_3d}_x', f'3d_{keypoint_id_3d}_y', f'3d_{keypoint_id_3d}_z']
        }
    elif feature_type == 'velocity':
        return {
            '2d_features': [f'vel_2d_{keypoint_id_2d}_x', f'vel_2d_{keypoint_id_2d}_y'],
            '3d_features': [f'vel_3d_{keypoint_id_3d}_x', f'vel_3d_{keypoint_id_3d}_y', f'vel_3d_{keypoint_id_3d}_z']
        }
    elif feature_type == 'acceleration':
        return {
            '2d_features': [f'acc_2d_{keypoint_id_2d}_x', f'acc_2d_{keypoint_id_2d}_y'],
            '3d_features': [f'acc_3d_{keypoint_id_3d}_x', f'acc_3d_{keypoint_id_3d}_y', f'acc_3d_{keypoint_id_3d}_z']
        }
    elif feature_type == 'angles':
        return {
            '2d_features': ['ang_2d_leftElbow', 'ang_2d_rightElbow'],
            '3d_features': ['ang_3d_leftElbow', 'ang_3d_rightElbow']
        }