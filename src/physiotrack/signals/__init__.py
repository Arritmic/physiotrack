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
)
from .plotting import RealTimePlotter, KeypointMotionPlotter
from .ppg import POS, CHROM, LGI, OMIT

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
    # plotting
    "RealTimePlotter",
    "KeypointMotionPlotter",
    # rPPG extraction
    "POS",
    "CHROM",
    "LGI",
    "OMIT",
]
