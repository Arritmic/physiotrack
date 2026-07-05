"""RR-interval artefact detection and correction (Lipponen & Tarvainen, 2019).

Detects and corrects ectopic, missed, extra and long/short beats in an RR-interval
(inter-beat-interval) series using the beat-classification algorithm of:

    Lipponen, J. A. & Tarvainen, M. P. (2019). "A robust algorithm for heart rate
    variability time series artefact correction using novel beat classification."
    *Journal of Medical Engineering & Technology* 43(3):173-181.

rPPG-derived RR series are noisy -- a single missed or spurious systolic peak throws
off RMSSD/SD1 dramatically -- so cleaning the series before HRV analysis is essential.
The algorithm classifies each beat from two decision subspaces built on the normalised
successive-difference series (``dRR``) and the deviation from a running median
(``mRR``), then repairs each class: extra beats are deleted, missed beats interpolated,
ectopic and long/short beats re-positioned to the local interval midpoint.

The classification thresholds (``c1 = 0.13``, ``c2 = 0.17``, ``alpha = 5.2``,
91-beat threshold window, 11-beat median window) are the paper's published values.
This is a native numpy/pandas re-implementation. It is validated in
``tests/test_artifacts.py`` against synthetic RR series with injected extra, missed and
ectopic beats (asserting each is detected and repaired, and that a clean series is left
unchanged), and additionally cross-checked against NeuroKit2's
``signal_fixpeaks(method="kubios")`` when NeuroKit2 is installed.
"""

import numpy as np
import pandas as pd

__all__ = ["find_rr_artifacts", "correct_rr_artifacts"]

# Published Lipponen & Tarvainen (2019) parameters.
_C1 = 0.13
_C2 = 0.17
_ALPHA = 5.2
_WINDOW_WIDTH = 91
_MEDFILT_ORDER = 11


def _compute_threshold(sig, alpha, window_width):
    """Quartile-deviation threshold over a centred rolling window (Eq. 1)."""
    df = pd.DataFrame({"signal": np.abs(sig)})
    q1 = df.rolling(window_width, center=True, min_periods=1).quantile(0.25).signal.values
    q3 = df.rolling(window_width, center=True, min_periods=1).quantile(0.75).signal.values
    return alpha * ((q3 - q1) / 2.0)


def _rr_from_peaks(peaks, sampling_rate):
    """Period series (seconds), same length as ``peaks`` with a realistic first value."""
    rr = np.ediff1d(peaks, to_begin=0) / sampling_rate
    if len(rr) > 1:
        rr[0] = np.mean(rr[1:])
    elif len(rr) == 1:
        rr[0] = 1.0
    return rr


def _find_artifacts(peaks, sampling_rate, c1=_C1, c2=_C2, alpha=_ALPHA,
                    window_width=_WINDOW_WIDTH, medfilt_order=_MEDFILT_ORDER):
    """Classify beats into ectopic / missed / extra / longshort (paper Figure 1).

    Faithful port of the Lipponen & Tarvainen decision flow (matching NeuroKit2's
    ``kubios`` implementation). Indices returned are positions in the period series.
    """
    rr = _rr_from_peaks(peaks, sampling_rate)

    # dRR: successive differences of the period series, threshold-normalised.
    drrs = np.ediff1d(rr, to_begin=0)
    if len(drrs) > 1:
        drrs[0] = np.mean(drrs[1:])
    elif len(drrs) == 1:
        drrs[0] = 0.0
    th1 = _compute_threshold(drrs, alpha, window_width)
    np.divide(drrs, th1, out=drrs, where=th1 != 0)
    drrs[th1 == 0] = np.nan

    # Decision subspaces s12 and s22 from the neighbouring dRR values.
    padding = 2
    drrs_pad = np.pad(drrs, padding, "reflect")
    s12 = np.zeros(drrs.size)
    s22 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):
        if drrs_pad[d] > 0:
            s12[d - padding] = np.max([drrs_pad[d - 1], drrs_pad[d + 1]])
        elif drrs_pad[d] < 0:
            s12[d - padding] = np.min([drrs_pad[d - 1], drrs_pad[d + 1]])
        if drrs_pad[d] >= 0:
            s22[d - padding] = np.min([drrs_pad[d + 1], drrs_pad[d + 2]])
        elif drrs_pad[d] < 0:
            s22[d - padding] = np.max([drrs_pad[d + 1], drrs_pad[d + 2]])

    # mRR: deviation of each period from a running median, threshold-normalised.
    medrr = (pd.DataFrame({"signal": rr})
             .rolling(medfilt_order, center=True, min_periods=1).median().signal.values)
    mrrs = rr - medrr
    mrrs[mrrs < 0] = mrrs[mrrs < 0] * 2
    th2 = _compute_threshold(mrrs, alpha, window_width)
    np.divide(mrrs, th2, out=mrrs, where=th2 != 0)
    mrrs[th2 == 0] = np.nan

    extra_idcs, missed_idcs, ectopic_idcs, longshort_idcs = [], [], [], []
    i = 0
    while i < rr.size - 2:
        if np.abs(drrs[i]) <= 1:
            i += 1
            continue
        eq1 = np.logical_and(drrs[i] > 1, s12[i] < (-c1 * drrs[i] - c2))
        eq2 = np.logical_and(drrs[i] < -1, s12[i] > (-c1 * drrs[i] + c2))
        if np.any([eq1, eq2]):
            ectopic_idcs.append(i)
            i += 1
            continue
        if ~np.any([np.abs(drrs[i]) > 1, np.abs(mrrs[i]) > 3]):
            i += 1
            continue
        longshort_candidates = [i]
        if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
            longshort_candidates.append(i + 1)
        for j in longshort_candidates:
            eq3 = np.logical_and(drrs[j] > 1, s22[j] < -1)
            eq4 = np.abs(mrrs[j]) > 3
            eq5 = np.logical_and(drrs[j] < -1, s22[j] > 1)
            if ~np.any([eq3, eq4, eq5]):
                i += 1
                continue
            eq6 = np.abs(rr[j] / 2 - medrr[j]) < th2[j]
            eq7 = np.abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]
            if np.all([eq5, eq7]):
                extra_idcs.append(j)
                i += 1
                continue
            if np.all([eq3, eq6]):
                missed_idcs.append(j)
                i += 1
                continue
            longshort_idcs.append(j)
            i += 1

    return {"ectopic": ectopic_idcs, "missed": missed_idcs,
            "extra": extra_idcs, "longshort": longshort_idcs}


def _update_indices(source_idcs, update_idcs, update):
    if not update_idcs:
        return update_idcs
    for s in source_idcs:
        update_idcs = [u + update if u > s else u for u in update_idcs]
    return list(np.unique(update_idcs))


def _correct_misaligned(misaligned_idcs, peaks):
    peaks = peaks.copy()
    idcs = np.array(misaligned_idcs)
    valid = np.logical_and(idcs > 1, idcs < len(peaks) - 1)
    idcs = idcs[valid]
    if idcs.size == 0:
        return peaks
    prev_peaks, next_peaks = peaks[idcs - 1], peaks[idcs + 1]
    interp = prev_peaks + (next_peaks - prev_peaks) / 2
    peaks = np.delete(peaks, idcs)
    peaks = np.concatenate((peaks, interp)).astype(int)
    peaks.sort(kind="mergesort")
    return peaks


def _correct_artifacts(artifacts, peaks):
    """Apply deletions/insertions/re-positioning, keeping index lists consistent."""
    extra, missed = artifacts["extra"], artifacts["missed"]
    ectopic, longshort = artifacts["ectopic"], artifacts["longshort"]
    if extra:
        peaks = np.delete(peaks.copy(), extra)
        missed = _update_indices(extra, missed, -1)
        ectopic = _update_indices(extra, ectopic, -1)
        longshort = _update_indices(extra, longshort, -1)
    if missed:
        idcs = np.array(missed)
        valid = np.logical_and(idcs > 1, idcs < len(peaks))
        idcs = idcs[valid]
        prev_peaks, next_peaks = peaks[idcs - 1], peaks[idcs]
        added = prev_peaks + (next_peaks - prev_peaks) / 2
        peaks = np.insert(peaks, idcs, added)
        ectopic = _update_indices(missed, ectopic, 1)
        longshort = _update_indices(missed, longshort, 1)
    if ectopic:
        peaks = _correct_misaligned(ectopic, peaks)
    if longshort:
        peaks = _correct_misaligned(longshort, peaks)
    return peaks


def _rri_to_peaks(rri_ms):
    """Reconstruct integer peak sample indices (1 sample == 1 ms) from an RR series."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(rri_ms, dtype=float))]).round().astype(int)


def find_rr_artifacts(rri_ms):
    """Classify each beat of an RR-interval series into artefact categories.

    Runs the Lipponen & Tarvainen (2019) beat classifier and returns the indices of
    each artefact class *within the internal period series* (position ``k`` refers to
    the ``k``-th beat). Use this for signal-quality reporting; use
    [`correct_rr_artifacts`][physiotrack.signals.correct_rr_artifacts] to also repair
    the series.

    Args:
        rri_ms (np.ndarray): RR-interval series in milliseconds (e.g. from
            [`bvp_to_rri`][physiotrack.signals.bvp_to_rri]).

    Returns:
        dict: ``{"ectopic": [...], "missed": [...], "extra": [...],
            "longshort": [...]}`` lists of integer beat indices.

    Example:
        ```python
        from physiotrack.signals import find_rr_artifacts
        art = find_rr_artifacts(rri_ms)
        print("ectopic beats:", art["ectopic"])
        ```
    """
    peaks = _rri_to_peaks(rri_ms)
    return _find_artifacts(peaks, sampling_rate=1000)


def correct_rr_artifacts(rri_ms, iterative=True):
    """Detect and correct artefacts in an RR-interval series (Lipponen-Tarvainen).

    Deletes extra beats, interpolates missed beats and re-positions ectopic /
    long-short beats to the local interval midpoint, returning a cleaned RR series
    ready for HRV analysis. Optionally repeats until no further artefacts are found.

    Args:
        rri_ms (np.ndarray): RR-interval series in milliseconds.
        iterative (bool, optional): Re-run detection+correction until the beat count
            stabilises (matching Kubios' iterative mode). Defaults to ``True``.

    Returns:
        tuple[np.ndarray, dict]: ``(rri_corrected_ms, artifacts)`` where
            ``rri_corrected_ms`` is the cleaned RR series in milliseconds (its length
            may differ from the input when beats are added or removed) and
            ``artifacts`` is the classification dict from the *first* pass (as returned
            by [`find_rr_artifacts`][physiotrack.signals.find_rr_artifacts]).

    Example:
        ```python
        from physiotrack.signals import bvp_to_rri, correct_rr_artifacts, compute_hrv
        rri_ms, _ = bvp_to_rri(bvp, fps=30.0)
        rri_clean, art = correct_rr_artifacts(rri_ms)
        hrv = compute_hrv(rri_clean)   # HRV on the cleaned series
        ```

    See Also:
        [`compute_hrv`][physiotrack.signals.compute_hrv]: consume the cleaned series.
    """
    rri = np.asarray(rri_ms, dtype=float)
    rri = rri[np.isfinite(rri)]
    if rri.size < 4:
        return rri, {"ectopic": [], "missed": [], "extra": [], "longshort": []}

    peaks = _rri_to_peaks(rri)
    first_artifacts = _find_artifacts(peaks, sampling_rate=1000)
    peaks = _correct_artifacts(first_artifacts, peaks)

    if iterative:
        prev = -1
        while peaks.size != prev:
            prev = peaks.size
            art = _find_artifacts(peaks, sampling_rate=1000)
            peaks = _correct_artifacts(art, peaks)

    rri_corrected = np.diff(np.sort(peaks)).astype(float)
    return rri_corrected, first_artifacts
