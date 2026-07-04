# Motion & Features

Turn keypoint results into motion signals: extract per-frame keypoint sequences, add
body/head/pelvic centroids, resample, and compute joint angles and range-of-motion
(ROM). See the [Signals guide](../../guides/signals.md).

## Keypoint sequences & centroids

::: physiotrack.signals.extract_keypoint_sequence_2d

::: physiotrack.signals.extract_keypoint_sequence_3d

::: physiotrack.signals.extract_keypoints_sequence

::: physiotrack.signals.add_head_centroid

::: physiotrack.signals.add_body_centroid

::: physiotrack.signals.add_pelvic_centroid

::: physiotrack.signals.resample_dataframe_by_interpolation

## Features, joint angles & ROM

::: physiotrack.signals.get_relative_coordinates

::: physiotrack.signals.compute_all_motion_features

::: physiotrack.signals.compute_all_joint_angles

::: physiotrack.signals.joint_angles

::: physiotrack.signals.compute_rom_angles

::: physiotrack.signals.get_keypoint_features

::: physiotrack.signals.select_feature_data
