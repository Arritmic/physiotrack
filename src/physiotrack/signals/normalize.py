import numpy as np
import pandas as pd


def min_max_normalize(series, feature_range=(0, 1)):
    """Min-max normalize a series to a target range.

    Linearly rescales values as
    ``(x - min) / (max - min) * (max_range - min_range) + min_range`` so the
    smallest value maps to ``min_range`` and the largest to ``max_range``. When
    the input is constant (``max == min``), returns the midpoint of the target
    range to avoid division by zero.

    Args:
        series (pd.Series): Input values to normalize.
        feature_range (tuple[float, float], optional): Target
            ``(min_range, max_range)`` output range. Defaults to ``(0, 1)``.

    Returns:
        pd.Series: Rescaled values spanning ``feature_range`` (constant input
            yields the range midpoint).

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import min_max_normalize
        scaled = min_max_normalize(pd.Series([1, 2, 3]), feature_range=(0, 1))
        ```
    """
    min_val = series.min()
    max_val = series.max()
    min_range, max_range = feature_range
    
    # Handle constant values
    if max_val == min_val:
        return pd.Series(np.full(len(series), (min_range + max_range) / 2))
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized * (max_range - min_range) + min_range


def z_score_normalize(series):
    """Z-score normalize (standardize) a series to zero mean and unit std.

    Computes ``(x - mean) / std`` so the result has mean 0 and standard
    deviation 1. When the input has zero variance (``std == 0``), returns all
    zeros to avoid division by zero.

    Args:
        series (pd.Series): Input values to standardize.

    Returns:
        pd.Series: Standardized values (mean 0, std 1); all zeros for constant
            input. Output is unbounded.

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import z_score_normalize
        z = z_score_normalize(pd.Series([1, 2, 3, 4]))
        ```
    """
    mean = series.mean()
    std = series.std()
    
    # Handle zero variance
    if std == 0:
        return pd.Series(np.zeros(len(series)))
    
    return (series - mean) / std


def robust_scale_normalize(series):
    """Robustly scale a series using the median and interquartile range.

    Computes ``(x - median) / IQR`` where ``IQR = Q3 - Q1`` (75th minus 25th
    percentile). Centering on the median and scaling by the IQR makes this far
    less sensitive to outliers than z-score normalization. When the IQR is zero,
    returns all zeros to avoid division by zero.

    Args:
        series (pd.Series): Input values to scale.

    Returns:
        pd.Series: Robustly scaled values (median maps to 0, IQR to 1); all
            zeros when the IQR is zero. Output is unbounded.

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import robust_scale_normalize
        scaled = robust_scale_normalize(pd.Series([1, 2, 3, 100]))
        ```

    See Also:
        [`z_score_normalize`][physiotrack.signals.z_score_normalize]:
            mean/std-based standardization.
    """
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    # Handle zero IQR
    if iqr == 0:
        return pd.Series(np.zeros(len(series)))
    
    return (series - median) / iqr


def max_abs_normalize(series):
    """Scale a series by its maximum absolute value.

    Divides every value by ``max(|x|)``, mapping the data into ``[-1, 1]`` while
    preserving zeros and sparsity (no centering). Returns the input unchanged
    when the maximum absolute value is zero.

    Args:
        series (pd.Series): Input values to scale.

    Returns:
        pd.Series: Values scaled into ``[-1, 1]`` (unchanged if all zeros).

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import max_abs_normalize
        scaled = max_abs_normalize(pd.Series([-2, 0, 4]))  # -> [-0.5, 0, 1]
        ```
    """
    max_abs = series.abs().max()
    
    if max_abs == 0:
        return series
    
    return series / max_abs


def decimal_scaling_normalize(series):
    """Normalize a series by decimal scaling.

    Divides every value by ``10 ** j``, where ``j`` is the number of digits in
    the integer part of the maximum absolute value, shifting the decimal point so
    the largest magnitude falls below 1. Returns the input unchanged when the
    maximum absolute value is zero.

    Args:
        series (pd.Series): Input values to scale.

    Returns:
        pd.Series: Decimal-scaled values with ``|x| < 1`` (unchanged if all
            zeros).

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import decimal_scaling_normalize
        scaled = decimal_scaling_normalize(pd.Series([120, 340, 990]))  # /1000
        ```
    """
    max_abs = series.abs().max()
    
    if max_abs == 0:
        return series
    
    # Find number of digits in max value
    j = len(str(int(max_abs)))
    return series / (10 ** j)


def log_normalize(series):
    """Apply a log(1 + x) transform to compress right-skewed data.

    Returns ``log1p(x) = ln(1 + x)``, which compresses large values and reduces
    right skew while mapping 0 to 0. Because ``1`` is added before the log,
    inputs down to ``-1`` (exclusive) are handled; values ``<= -1`` produce
    ``nan``/``-inf``.

    Args:
        series (pd.Series): Input values, expected to be ``> -1``.

    Returns:
        pd.Series: The element-wise ``ln(1 + x)`` transform.

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import log_normalize
        out = log_normalize(pd.Series([0, 9, 99]))  # -> [0, ~2.30, ~4.61]
        ```
    """
    # Add small constant to handle zeros
    return np.log1p(series)


def sigmoid_normalize(series):
    """Map a series through the logistic sigmoid into (0, 1).

    Applies ``1 / (1 + exp(-x))`` element-wise, squashing any real value into the
    open interval ``(0, 1)`` with 0 mapping to 0.5. Inputs are used as-is (no
    prior standardization), so pre-scaling is recommended for well-spread output.

    Args:
        series (pd.Series): Input values (any real range).

    Returns:
        pd.Series: Values in the open interval ``(0, 1)``.

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import sigmoid_normalize
        out = sigmoid_normalize(pd.Series([-1, 0, 1]))
        ```
    """
    return 1 / (1 + np.exp(-series))


def tanh_normalize(series):
    """Map a series through the hyperbolic tangent into (-1, 1).

    Applies ``tanh(x)`` element-wise, squashing any real value into the open
    interval ``(-1, 1)`` with 0 mapping to 0. Inputs are used as-is (no prior
    standardization).

    Args:
        series (pd.Series): Input values (any real range).

    Returns:
        pd.Series: Values in the open interval ``(-1, 1)``.

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import tanh_normalize
        out = tanh_normalize(pd.Series([-2, 0, 2]))
        ```
    """
    return np.tanh(series)


def unit_vector_normalize(series):
    """Scale a series to unit L2 norm.

    Divides the series by its Euclidean (L2) norm so the resulting vector has
    length 1, preserving direction while discarding magnitude. Returns the input
    unchanged when the norm is zero.

    Args:
        series (pd.Series): Input vector to normalize.

    Returns:
        pd.Series: The unit-norm vector (unchanged if the input norm is zero).

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import unit_vector_normalize
        u = unit_vector_normalize(pd.Series([3.0, 4.0]))  # -> [0.6, 0.8]
        ```
    """
    norm = np.linalg.norm(series)
    
    if norm == 0:
        return series
    
    return series / norm


def quantile_normalize(series, n_quantiles=100):
    """Map a series to a uniform distribution via quantile transformation.

    Uses ``sklearn.preprocessing.QuantileTransformer`` with
    ``output_distribution="uniform"``, fitted on the input itself, to remap
    values onto a uniform distribution in ``[0, 1]``. This is a rank-based
    (non-linear, monotonic) transform that flattens the value distribution and
    is robust to outliers.

    Args:
        series (pd.Series): Input values to transform.
        n_quantiles (int, optional): Number of quantiles used to estimate the
            distribution. Defaults to ``100``.

    Returns:
        pd.Series: Values mapped onto a uniform distribution in ``[0, 1]`` (index
            reset, as it is rebuilt from the flattened transform output).

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import quantile_normalize
        out = quantile_normalize(pd.Series(range(200)), n_quantiles=50)
        ```

    Note:
        Requires scikit-learn. ``n_quantiles`` larger than the number of samples
        is clipped by scikit-learn with a warning.
    """
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    return pd.Series(qt.fit_transform(series.values.reshape(-1, 1)).flatten())


def power_transform_normalize(series, method='yeo-johnson'):
    """Apply a power transform to make a series more Gaussian-like.

    Uses ``sklearn.preprocessing.PowerTransformer``, fitted on the input, to
    stabilize variance and reduce skew. ``"yeo-johnson"`` supports zero and
    negative values; ``"box-cox"`` requires strictly positive input. The output
    is additionally standardized to zero mean and unit variance by the
    transformer.

    Args:
        series (pd.Series): Input values to transform.
        method (str, optional): Power transform family, either
            ``"yeo-johnson"`` or ``"box-cox"``. Defaults to ``"yeo-johnson"``.

    Returns:
        pd.Series: The transformed, standardized values (index reset, rebuilt
            from the flattened transform output).

    Example:
        ```python
        import pandas as pd
        from physiotrack.signals.normalize import power_transform_normalize
        out = power_transform_normalize(pd.Series([1, 2, 4, 8, 16]))
        ```

    Raises:
        ValueError: If ``method="box-cox"`` is used with non-positive values
            (raised by scikit-learn).

    Note:
        Requires scikit-learn.
    """
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method=method)
    return pd.Series(pt.fit_transform(series.values.reshape(-1, 1)).flatten())