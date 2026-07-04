"""
Public API for the physiotrack signals subsystem.

Re-exports the key signal-processing functions and classes so users can do::

    from physiotrack.signals import bandpass_filter, POS, RealTimePlotter
"""

from .filters import (
    bandpass_filter,
    zero_mean_std_norm,
    zero_mean_std_norm_1ch,
    band_pass_filter,
    notch_filter,
    highpass_filter,
    lowpass_filter,
    detrend_advanced,
    bandpass_firwin,
    signaltonoise_dB,
)
from .normalize import (
    min_max_normalize,
    z_score_normalize,
    robust_scale_normalize,
    max_abs_normalize,
    decimal_scaling_normalize,
    log_normalize,
    sigmoid_normalize,
    tanh_normalize,
    unit_vector_normalize,
    quantile_normalize,
    power_transform_normalize,
)
from .evaluate import (
    compute_plv,
    event_synchronization,
    phase_synchrony,
    compute_rmse,
    align_signals,
    normalized_cross_correlation,
    calculate_pearson_correlation,
    calculate_dtw_distance,
    hrv_errors,
)
from .plotting import (
    RealTimePlotter,
    KeypointMotionPlotter,
    JointAnglePlotter,
    HeartRatePlotter,
    RPPGPlotter,
    HRVPlotter,
    RespirationPlotter,
)
from .ppg import POS, CHROM, LGI, OMIT
from .ppg.constants import (
    HR_BAND, RESP_BAND, HRV_VLF_BAND, HRV_LF_BAND, HRV_HF_BAND,
    RPPG_METHODS, DEFAULT_RPPG_METHOD,
)
from .ppg.metrics import bvp_to_hr, bvp_snr, hr_errors, benchmark_rppg_methods
from .ppg.estimator import HeartRateEstimator
from .ppg.skin import FaceSkinExtractor, FaceParsing
from .ppg.peaks import detect_pulse_peaks, bvp_to_rri
from .ppg.artifacts import find_rr_artifacts, correct_rr_artifacts
from .ppg.hrv import (
    hrv_time, hrv_frequency, hrv_nonlinear, compute_hrv,
    sample_entropy, approximate_entropy,
)
from .ppg.respiration import (
    respiration_rate_from_signal, respiration_from_pulse, respiration_from_rri,
)
from .motion.utils import (
    extract_keypoint_sequence_2d,
    extract_keypoint_sequence_3d,
    extract_keypoints_sequence,
    add_head_centroid,
    add_body_centroid,
    add_pelvic_centroid,
    resample_dataframe_by_interpolation,
)
from .motion.features import (
    get_relative_coordinates,
    compute_all_motion_features,
    compute_all_joint_angles,
    joint_angles,
    compute_rom_angles,
    get_keypoint_features,
    select_feature_data,
    respiration_from_motion,
)

__all__ = [
    # filters
    "bandpass_filter",
    "zero_mean_std_norm",
    "zero_mean_std_norm_1ch",
    "band_pass_filter",
    "notch_filter",
    "highpass_filter",
    "lowpass_filter",
    "detrend_advanced",
    "bandpass_firwin",
    "signaltonoise_dB",
    # normalize
    "min_max_normalize",
    "z_score_normalize",
    "robust_scale_normalize",
    "max_abs_normalize",
    "decimal_scaling_normalize",
    "log_normalize",
    "sigmoid_normalize",
    "tanh_normalize",
    "unit_vector_normalize",
    "quantile_normalize",
    "power_transform_normalize",
    # evaluate
    "compute_plv",
    "event_synchronization",
    "phase_synchrony",
    "compute_rmse",
    "align_signals",
    "normalized_cross_correlation",
    "calculate_pearson_correlation",
    "calculate_dtw_distance",
    "hrv_errors",
    # plotting
    "RealTimePlotter",
    "KeypointMotionPlotter",
    "JointAnglePlotter",
    "HeartRatePlotter",
    "RPPGPlotter",
    "HRVPlotter",
    "RespirationPlotter",
    # rPPG extraction
    "POS",
    "CHROM",
    "LGI",
    "OMIT",
    # rPPG analysis bands + method registry (single source of truth)
    "HR_BAND",
    "RESP_BAND",
    "HRV_VLF_BAND",
    "HRV_LF_BAND",
    "HRV_HF_BAND",
    "RPPG_METHODS",
    "DEFAULT_RPPG_METHOD",
    # rPPG HR metrics + estimator + skin extraction
    "bvp_to_hr",
    "bvp_snr",
    "hr_errors",
    "benchmark_rppg_methods",
    "HeartRateEstimator",
    "FaceSkinExtractor",
    "FaceParsing",
    # RR-interval extraction, artefact correction, HRV, respiration
    "detect_pulse_peaks",
    "bvp_to_rri",
    "find_rr_artifacts",
    "correct_rr_artifacts",
    "hrv_time",
    "hrv_frequency",
    "hrv_nonlinear",
    "compute_hrv",
    "sample_entropy",
    "approximate_entropy",
    "respiration_rate_from_signal",
    "respiration_from_pulse",
    "respiration_from_rri",
    "respiration_from_motion",
    # motion: keypoint sequences & centroids
    "extract_keypoint_sequence_2d",
    "extract_keypoint_sequence_3d",
    "extract_keypoints_sequence",
    "add_head_centroid",
    "add_body_centroid",
    "add_pelvic_centroid",
    "resample_dataframe_by_interpolation",
    # motion: features
    "get_relative_coordinates",
    "compute_all_motion_features",
    "compute_all_joint_angles",
    "joint_angles",
    "compute_rom_angles",
    "get_keypoint_features",
    "select_feature_data",
]
