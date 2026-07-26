from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.signal import hilbert
from scipy.signal import find_peaks
import warnings


def compute_plv(signal1, signal2):
    """Compute the Phase Locking Value (PLV) between two 1D signals.

    Trims both signals to equal length via
    [`align_signals`][physiotrack.signals.align_signals], extracts the
    instantaneous phase of each via the Hilbert transform, and measures how
    consistent their phase difference is: ``|mean(exp(1j * (phase1 - phase2)))|``.
    A value near 1 means the two signals stay phase-locked; near 0 means their
    phase relationship is random.

    Args:
        signal1 (np.ndarray): First 1D signal (e.g., IMU-based wrist motion).
        signal2 (np.ndarray): Second 1D signal (e.g., video-based wrist motion).

    Returns:
        float: PLV in ``[0, 1]``, where 1 indicates perfect phase
            synchronization.

    Example:
        ```python
        from physiotrack.signals.evaluate import compute_plv
        plv = compute_plv(imu_trace, video_trace)
        ```

    See Also:
        [`phase_synchrony`][physiotrack.signals.phase_synchrony]: alternative
            phase-agreement metric based on mean absolute phase difference.
    """
    # Compute the analytic signal using Hilbert Transform
    signal1, signal2 = align_signals(signal1, signal2)
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract instantaneous phase angles
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    # Compute the phase difference
    phase_diff = phase1 - phase2

    # Compute PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))  # Mean of complex exponentials

    return plv


def event_synchronization(signal1, signal2, max_delay=5):
    """Compute the Event Synchronization Index (ESI) between two signals.

    Trims both signals to equal length, detects peaks in each (with a minimum
    peak separation of 40 samples), and counts how many peaks in ``signal1``
    have a matching peak in ``signal2`` within ``max_delay`` samples. The count
    is normalized by the larger peak count of the two signals.

    Args:
        signal1 (np.ndarray): First 1D signal.
        signal2 (np.ndarray): Second 1D signal.
        max_delay (int, optional): Maximum sample offset for two peaks to be
            considered synchronized. Defaults to ``5``.

    Returns:
        float: ESI in ``[0, 1]``, where 1 means every peak has a synchronized
            counterpart.

    Example:
        ```python
        from physiotrack.signals.evaluate import event_synchronization
        esi = event_synchronization(sig_a, sig_b, max_delay=5)
        ```

    Note:
        Peak detection uses a fixed minimum inter-peak distance of 40 samples,
        so the metric is sensitive to sampling rate.
    """
    signal1, signal2 = align_signals(signal1, signal2)
    peaks1, _ = find_peaks(signal1, distance=40)  # Detect peaks
    peaks2, _ = find_peaks(signal2, distance=40)

    denom = max(len(peaks1), len(peaks2))
    if denom == 0:
        # Neither signal has a detectable peak: no events to synchronise.
        return 0.0

    count = 0
    for p1 in peaks1:
        if any(abs(p1 - p2) <= max_delay for p2 in peaks2):
            count += 1

    esi = count / denom  # Normalize
    return esi


def phase_synchrony(signal1, signal2):
    """Compute phase synchronization from the mean absolute phase difference.

    Trims both signals to equal length, extracts each signal's instantaneous
    phase via the Hilbert transform, and returns
    ``1 - mean(|phase1 - phase2|) / pi``. A value near 1 means the phases stay
    close; near 0 means they are maximally out of step.

    Args:
        signal1 (np.ndarray): First 1D signal.
        signal2 (np.ndarray): Second 1D signal.

    Returns:
        float: Phase-synchrony index, nominally in ``[0, 1]`` (1 = fully
            synchronized).

    Example:
        ```python
        from physiotrack.signals.evaluate import phase_synchrony
        ps = phase_synchrony(sig_a, sig_b)
        ```

    See Also:
        [`compute_plv`][physiotrack.signals.compute_plv]: phase-locking value,
            a complementary phase-agreement metric.
    """
    signal1, signal2 = align_signals(signal1, signal2)
    phase1 = np.angle(hilbert(signal1))  # Extract phase
    phase2 = np.angle(hilbert(signal2))

    # Wrap the phase difference to (-pi, pi] before taking the magnitude, so the
    # circular distance is measured correctly (a raw phase1 - phase2 spans
    # [-2pi, 2pi], which would let the index fall below 0 for in-phase signals
    # whose instantaneous phases straddle the +/-pi wrap-around).
    phase_diff = np.abs(np.angle(np.exp(1j * (phase1 - phase2))))
    return 1 - (np.mean(phase_diff) / np.pi)  # Normalize between 0 and 1


def compute_rmse(signal1, signal2):
    """Compute the root-mean-square error (RMSE) between two signals.

    Trims both signals to equal length and returns the square root of their
    mean squared error. RMSE is in the same units as the input signals; lower is
    better, 0 means identical.

    Args:
        signal1 (np.ndarray): First 1D signal.
        signal2 (np.ndarray): Second 1D signal.

    Returns:
        float: RMSE between the two signals (``>= 0``).

    Example:
        ```python
        from physiotrack.signals.evaluate import compute_rmse
        err = compute_rmse(reference, estimate)
        ```
    """
    # Align signals to same length
    signal1, signal2 = align_signals(signal1, signal2)
    return np.sqrt(mean_squared_error(signal1, signal2))


def align_signals(signal1, signal2):
    """Trim two signals to a common length.

    Truncates both signals to the length of the shorter one (keeping the leading
    samples). Used internally by the other metrics in this module to guarantee
    equal-length inputs.

    Args:
        signal1 (np.ndarray): First 1D signal.
        signal2 (np.ndarray): Second 1D signal.

    Returns:
        tuple[np.ndarray, np.ndarray]: The two signals, each truncated to
            ``min(len(signal1), len(signal2))`` samples.

    Example:
        ```python
        from physiotrack.signals.evaluate import align_signals
        a, b = align_signals(long_signal, short_signal)
        ```
    """
    min_length = min(len(signal1), len(signal2))
    return signal1[:min_length], signal2[:min_length]


def normalized_cross_correlation(signal1, signal2):
    """Compute the zero-lag normalized cross-correlation between two signals.

    Trims both signals to equal length, z-score normalizes each (subtract mean,
    divide by std), then returns their zero-lag cross-correlation divided by the
    signal length. For matched signals this approximates the Pearson correlation
    at lag 0, so it falls in ``[-1, 1]``.

    Args:
        signal1 (np.ndarray): First 1D signal.
        signal2 (np.ndarray): Second 1D signal.

    Returns:
        float: Normalized zero-lag cross-correlation in ``[-1, 1]`` (1 = perfect
            positive correlation).

    Example:
        ```python
        from physiotrack.signals.evaluate import normalized_cross_correlation
        ncc = normalized_cross_correlation(sig_a, sig_b)
        ```

    See Also:
        [`calculate_pearson_correlation`][physiotrack.signals.calculate_pearson_correlation]:
            Pearson correlation with constant-signal handling.
    """
    # Align signals to same length
    signal1, signal2 = align_signals(signal1, signal2)
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    return np.correlate(signal1, signal2, mode="valid")[0] / len(signal1)


def calculate_pearson_correlation(signal1, signal2):
    """Compute the Pearson correlation coefficient between two signals.

    Trims both signals to equal length, z-score normalizes each, and returns
    their Pearson correlation. If either signal is constant (correlation
    undefined) or a ``ValueError`` occurs, prints a warning and returns
    ``np.nan``.

    Args:
        signal1 (np.ndarray): First 1D signal.
        signal2 (np.ndarray): Second 1D signal.

    Returns:
        float: Pearson correlation coefficient in ``[-1, 1]``, or ``np.nan`` if
            it cannot be computed (e.g. a constant signal).

    Example:
        ```python
        from physiotrack.signals.evaluate import calculate_pearson_correlation
        r = calculate_pearson_correlation(reference, estimate)
        ```

    See Also:
        [`normalized_cross_correlation`][physiotrack.signals.normalized_cross_correlation]:
            zero-lag normalized cross-correlation.
    """
    try:
        # Align signals to same length
        signal1, signal2 = align_signals(signal1, signal2)

        # Check for constant signals
        if np.all(signal1 == signal1[0]) or np.all(signal2 == signal2[0]):
            warnings.warn("One of the signals is constant, so Pearson correlation is "
                          "undefined; returning nan.", RuntimeWarning, stacklevel=2)
            return np.nan

        signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
        signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)

        correlation, _ = pearsonr(signal1, signal2)
        return correlation
    except ValueError:
        warnings.warn("Could not compute Pearson correlation; the signals may be "
                      "invalid. Returning nan.", RuntimeWarning, stacklevel=2)
        return np.nan


def calculate_dtw_distance(signal1, signal2, distance_metric=euclidean):
    """Compute the Dynamic Time Warping (DTW) distance between two signals.

    Trims both signals to equal length and computes their DTW distance with
    ``fastdtw``, which finds the optimal non-linear temporal alignment and is
    robust to time shifts and local speed differences. On any error, prints a
    message and returns ``np.nan``.

    Args:
        signal1 (np.ndarray): First 1D signal.
        signal2 (np.ndarray): Second 1D signal.
        distance_metric (callable, optional): Point-wise distance function passed
            to ``fastdtw``. Defaults to ``scipy.spatial.distance.euclidean``.

    Returns:
        float: The DTW distance (``>= 0``; lower means more similar), or
            ``np.nan`` if it cannot be computed.

    Example:
        ```python
        from physiotrack.signals.evaluate import calculate_dtw_distance
        dist = calculate_dtw_distance(reference, estimate)
        ```

    Note:
        Requires the ``fastdtw`` package.
    """
    try:
        # Align signals before computing DTW
        signal1, signal2 = align_signals(signal1, signal2)
        distance, _ = fastdtw(signal1.reshape(-1, 1), signal2.reshape(-1, 1), dist=distance_metric)
        return distance
    except Exception as e:
        warnings.warn(f"Could not compute the DTW distance: {e!r}. Returning nan.",
                      RuntimeWarning, stacklevel=2)
        return np.nan


def hrv_errors(hrv_est, hrv_ref, keys=None):
    """Per-metric agreement between an estimated and a reference HRV index set.

    Compares two HRV dicts (as returned by
    [`compute_hrv`][physiotrack.signals.compute_hrv]) key-by-key, reporting the signed
    difference and the percentage error relative to the reference. The rPPG-vs-contact
    counterpart of [`hr_errors`][physiotrack.signals.hr_errors] for validating
    contactless HRV against a ground-truth device.

    Args:
        hrv_est (dict): Estimated HRV indices (e.g. from rPPG-derived RR intervals).
        hrv_ref (dict): Reference HRV indices (e.g. from a contact ECG/PPG device).
        keys (Sequence[str], optional): Metrics to compare. Defaults to the intersection
            of the two dicts' keys.

    Returns:
        dict: ``{metric: {"est": .., "ref": .., "diff": est-ref, "pct": percent_error}}``
            for each compared metric; entries with a missing or non-finite value are
            skipped.

    Example:
        ```python
        from physiotrack.signals import compute_hrv, hrv_errors
        err = hrv_errors(compute_hrv(rri_rppg), compute_hrv(rri_ecg))
        print(err["RMSSD"]["pct"])
        ```
    """
    if keys is None:
        keys = [k for k in hrv_est if k in hrv_ref]
    out = {}
    for k in keys:
        e, r = hrv_est.get(k), hrv_ref.get(k)
        if not (isinstance(e, (int, float)) and isinstance(r, (int, float))):
            continue
        if not (np.isfinite(e) and np.isfinite(r)):
            continue
        diff = float(e) - float(r)
        out[k] = {"est": float(e), "ref": float(r), "diff": diff,
                  "pct": float(abs(diff) / abs(r) * 100.0) if r != 0 else np.nan}
    return out