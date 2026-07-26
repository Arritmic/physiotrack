"""Heart-rate-variability (HRV) metrics from an RR-interval series.

Computes the standard time-domain, frequency-domain and non-linear HRV indices from
a series of RR intervals (inter-beat intervals, in milliseconds) such as the one
produced by [`bvp_to_rri`][physiotrack.signals.bvp_to_rri]. Definitions follow:

* **Time / frequency domain** -- Task Force of the European Society of Cardiology and
  the North American Society of Pacing and Electrophysiology, "Heart rate variability:
  standards of measurement, physiological interpretation and clinical use",
  *Circulation* 93(5):1043-1065, 1996.
* **Poincare (SD1/SD2)** -- Brennan, Palaniswami & Kamen, "Do existing measures of
  Poincare plot geometry reflect nonlinear features of HRV?", *IEEE TBME* 48(11), 2001.
* **Sample / approximate entropy** -- Richman & Moorman, "Physiological time-series
  analysis using approximate entropy and sample entropy", *Am. J. Physiol.* 278(6),
  2000; Pincus, *PNAS* 88, 1991.

The implementations here are original (numpy/scipy). They are validated in
``tests/test_hrv.py`` against closed-form references on known RR series (RMSSD/SDNN/
pNNx by direct formula, SD1/SD2 via the Poincare identities, sample/approximate entropy
against a brute-force reference, and band powers on a synthetic tachogram with a known
LF/HF tone), and additionally cross-checked against NeuroKit2 when it is installed.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.interpolate import interp1d

from physiotrack.signals.ppg.constants import HRV_VLF_BAND, HRV_LF_BAND, HRV_HF_BAND

__all__ = [
    "hrv_time",
    "hrv_frequency",
    "hrv_nonlinear",
    "compute_hrv",
    "sample_entropy",
    "approximate_entropy",
]


def hrv_time(rri_ms):
    """Time-domain HRV indices from an RR-interval series.

    Args:
        rri_ms (np.ndarray): RR-interval (inter-beat-interval) series in milliseconds.

    Returns:
        dict: Time-domain indices. Keys:

            * ``MeanNN`` -- mean RR interval (ms).
            * ``SDNN`` -- standard deviation of RR intervals (ms, sample std, ddof=1).
            * ``RMSSD`` -- root mean square of successive RR differences (ms).
            * ``SDSD`` -- standard deviation of successive differences (ms, ddof=1).
            * ``pNN50`` / ``pNN20`` -- count of successive RR differences greater than
              50 / 20 ms, as a percentage of the total number of NN intervals.
            * ``CVNN`` -- coefficient of variation (``SDNN / MeanNN``).
            * ``MedianNN`` -- median RR interval (ms).
            * ``MeanHR`` / ``SDHR`` / ``MinHR`` / ``MaxHR`` -- instantaneous heart-rate
              statistics in bpm (``60000 / RR``).

    Note:
        All values are ``np.nan`` if fewer than 2 intervals are supplied.

    Example:
        ```python
        from physiotrack.signals.ppg.hrv import hrv_time
        idx = hrv_time(rri_ms)
        print(idx["RMSSD"], idx["SDNN"])
        ```
    """
    rri = np.asarray(rri_ms, dtype=float)
    rri = rri[np.isfinite(rri)]
    if rri.size < 2:
        return {k: np.nan for k in ("MeanNN", "SDNN", "RMSSD", "SDSD", "pNN50",
                                    "pNN20", "CVNN", "MedianNN", "MeanHR", "SDHR",
                                    "MinHR", "MaxHR")}
    diff = np.diff(rri)
    sdnn = float(np.std(rri, ddof=1))
    mean_nn = float(np.mean(rri))
    hr = 60000.0 / rri
    return {
        "MeanNN": mean_nn,
        "SDNN": sdnn,
        "RMSSD": float(np.sqrt(np.mean(diff ** 2))),
        "SDSD": float(np.std(diff, ddof=1)),
        # pNNx: NNx count over the total number of NN intervals (Task Force 1996).
        "pNN50": float(100.0 * np.sum(np.abs(diff) > 50.0) / rri.size),
        "pNN20": float(100.0 * np.sum(np.abs(diff) > 20.0) / rri.size),
        "CVNN": float(sdnn / mean_nn) if mean_nn else np.nan,
        "MedianNN": float(np.median(rri)),
        "MeanHR": float(np.mean(hr)),
        "SDHR": float(np.std(hr, ddof=1)),
        "MinHR": float(np.min(hr)),
        "MaxHR": float(np.max(hr)),
    }


def hrv_frequency(rri_ms, t_sec=None, interpolation_rate=4.0,
                  vlf_band=HRV_VLF_BAND, lf_band=HRV_LF_BAND, hf_band=HRV_HF_BAND):
    """Frequency-domain HRV indices via Welch PSD of the interpolated tachogram.

    The RR series is resampled onto an evenly-spaced grid (cubic interpolation, default
    4 Hz per Task Force practice), mean-removed, and its power spectral density is
    estimated with Welch's method. Band powers are the trapezoidal integral of the PSD
    over each band, in ``ms^2``.

    Args:
        rri_ms (np.ndarray): RR-interval series in milliseconds.
        t_sec (np.ndarray, optional): Beat timestamps in seconds, one per interval. If
            ``None``, cumulative RR times are used. Defaults to ``None``.
        interpolation_rate (float, optional): Resampling rate of the tachogram in Hz.
            Defaults to ``4.0``.
        vlf_band (tuple[float, float], optional): VLF band edges in Hz. Defaults to
            [`HRV_VLF_BAND`][physiotrack.signals.ppg.constants.HRV_VLF_BAND].
        lf_band (tuple[float, float], optional): LF band edges in Hz. Defaults to
            [`HRV_LF_BAND`][physiotrack.signals.ppg.constants.HRV_LF_BAND].
        hf_band (tuple[float, float], optional): HF band edges in Hz. Defaults to
            [`HRV_HF_BAND`][physiotrack.signals.ppg.constants.HRV_HF_BAND].

    Returns:
        dict: Frequency-domain indices. Keys ``VLF``, ``LF``, ``HF`` (absolute power,
            ms^2), ``TotalPower`` (ms^2), ``LFHF`` (``LF / HF`` ratio), ``LFn`` /
            ``HFn`` (normalized units, ``LF`` or ``HF`` over ``LF + HF``, in percent).
            All ``np.nan`` if fewer than 4 intervals are supplied.

    Example:
        ```python
        from physiotrack.signals.ppg.hrv import hrv_frequency
        f = hrv_frequency(rri_ms, t_sec)
        print(f["LF"], f["HF"], f["LFHF"])
        ```

    Note:
        Frequency-domain HRV needs a reasonably long recording (Task Force recommends
        >= ~2 min for LF and >= ~1 min for HF); short windows give unstable estimates.
    """
    keys = ("VLF", "LF", "HF", "TotalPower", "LFHF", "LFn", "HFn")
    rri = np.asarray(rri_ms, dtype=float)
    if rri.size < 4:
        return {k: np.nan for k in keys}
    if t_sec is None:
        t = np.cumsum(rri) / 1000.0
    else:
        t = np.asarray(t_sec, dtype=float)
    m = np.isfinite(rri) & np.isfinite(t)
    rri, t = rri[m], t[m]
    if rri.size < 4:
        return {k: np.nan for k in keys}

    fs = float(interpolation_rate)
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    if t_uniform.size < 4:
        return {k: np.nan for k in keys}
    kind = "cubic" if rri.size >= 4 else "linear"
    tachogram = interp1d(t, rri, kind=kind, fill_value="extrapolate")(t_uniform)
    tachogram = tachogram - np.mean(tachogram)

    nperseg = min(len(tachogram), int(fs * 60))  # ~60 s segments, capped by length
    freqs, psd = welch(tachogram, fs=fs, window="hann",
                       nperseg=max(4, nperseg), detrend="constant")

    def _bandpower(band):
        sel = (freqs >= band[0]) & (freqs < band[1])
        return float(trapezoid(psd[sel], freqs[sel])) if np.any(sel) else 0.0

    vlf, lf, hf = _bandpower(vlf_band), _bandpower(lf_band), _bandpower(hf_band)
    total = vlf + lf + hf
    lf_hf = lf / hf if hf > 0 else np.nan
    denom = lf + hf
    return {
        "VLF": vlf, "LF": lf, "HF": hf, "TotalPower": total, "LFHF": lf_hf,
        "LFn": float(100.0 * lf / denom) if denom > 0 else np.nan,
        "HFn": float(100.0 * hf / denom) if denom > 0 else np.nan,
    }


def _embed(x, m):
    """Stack length-``m`` delay vectors of ``x`` as rows of an ``(N-m+1, m)`` array."""
    n = len(x) - m + 1
    return np.array([x[i:i + m] for i in range(n)]) if n > 0 else np.empty((0, m))


def _chebyshev_matches(vectors, r):
    """Total count of ordered vector pairs (i != j) within Chebyshev distance ``r``."""
    count = 0
    for i in range(len(vectors)):
        d = np.max(np.abs(vectors - vectors[i]), axis=1)
        count += int(np.sum(d <= r)) - 1  # exclude the self-match
    return count


def sample_entropy(x, m=2, r=None):
    """Sample entropy (SampEn) of a 1-D series (Richman & Moorman, 2000).

    ``SampEn = -ln(A / B)`` where ``B`` is the number of ordered pairs of length-``m``
    template vectors within Chebyshev distance ``r`` (excluding self-matches) and ``A``
    is the same count for length ``m + 1``. Unlike approximate entropy, self-matches
    are excluded, making SampEn largely independent of series length and less biased.

    Args:
        x (np.ndarray): 1-D input series (e.g. an RR-interval series).
        m (int, optional): Embedding dimension (template length). Defaults to ``2``.
        r (float, optional): Tolerance for matches. If ``None``, ``0.2 * std(x)``
            (sample std, ddof=1) is used, the common HRV convention. Defaults to
            ``None``.

    Returns:
        float: The sample entropy, or ``np.nan`` if it is undefined (too few points, or
            no length-``m+1`` matches).

    Example:
        ```python
        from physiotrack.signals.ppg.hrv import sample_entropy
        se = sample_entropy(rri_ms, m=2)
        ```
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r <= 0:
        return np.nan
    b = _chebyshev_matches(_embed(x, m), r)
    a = _chebyshev_matches(_embed(x, m + 1), r)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def approximate_entropy(x, m=2, r=None):
    """Approximate entropy (ApEn) of a 1-D series (Pincus, 1991).

    ``ApEn = phi_m - phi_{m+1}`` where ``phi_k`` is the mean over template vectors of
    the log fraction of length-``k`` vectors within Chebyshev distance ``r`` (self-
    matches *included*, unlike sample entropy).

    Args:
        x (np.ndarray): 1-D input series.
        m (int, optional): Embedding dimension. Defaults to ``2``.
        r (float, optional): Tolerance; defaults to ``0.2 * std(x)`` (ddof=1) when
            ``None``. Defaults to ``None``.

    Returns:
        float: The approximate entropy, or ``np.nan`` if undefined.

    Example:
        ```python
        from physiotrack.signals.ppg.hrv import approximate_entropy
        ap = approximate_entropy(rri_ms, m=2)
        ```
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r <= 0:
        return np.nan

    def _phi(k):
        vectors = _embed(x, k)
        nv = len(vectors)
        c = np.empty(nv)
        for i in range(nv):
            d = np.max(np.abs(vectors - vectors[i]), axis=1)
            c[i] = np.sum(d <= r) / nv  # self-match included
        return float(np.mean(np.log(c)))

    return float(_phi(m) - _phi(m + 1))


def hrv_nonlinear(rri_ms):
    """Non-linear HRV indices (Poincare geometry + entropy).

    Args:
        rri_ms (np.ndarray): RR-interval series in milliseconds.

    Returns:
        dict: Non-linear indices. Keys:

            * ``SD1`` -- short-term Poincare dispersion (ms), ``sqrt(0.5) * SDSD``.
            * ``SD2`` -- long-term Poincare dispersion (ms),
              ``sqrt(2*SDNN^2 - 0.5*SDSD^2)``.
            * ``SD1SD2`` -- ratio ``SD1 / SD2``.
            * ``S`` -- Poincare ellipse area, ``pi * SD1 * SD2``.
            * ``SampEn`` -- sample entropy (``m = 2``).
            * ``ApEn`` -- approximate entropy (``m = 2``).

    Note:
        All values are ``np.nan`` if fewer than 3 intervals are supplied.

    Example:
        ```python
        from physiotrack.signals.ppg.hrv import hrv_nonlinear
        nl = hrv_nonlinear(rri_ms)
        print(nl["SD1"], nl["SD2"], nl["SampEn"])
        ```
    """
    keys = ("SD1", "SD2", "SD1SD2", "S", "SampEn", "ApEn")
    rri = np.asarray(rri_ms, dtype=float)
    rri = rri[np.isfinite(rri)]
    if rri.size < 3:
        return {k: np.nan for k in keys}
    sdnn = np.std(rri, ddof=1)
    sdsd = np.std(np.diff(rri), ddof=1)
    sd1 = np.sqrt(0.5) * sdsd
    sd2 = np.sqrt(max(2.0 * sdnn ** 2 - 0.5 * sdsd ** 2, 0.0))
    return {
        "SD1": float(sd1),
        "SD2": float(sd2),
        "SD1SD2": float(sd1 / sd2) if sd2 > 0 else np.nan,
        "S": float(np.pi * sd1 * sd2),
        "SampEn": sample_entropy(rri, m=2),
        "ApEn": approximate_entropy(rri, m=2),
    }


def compute_hrv(rri_ms, t_sec=None, domains=("time", "frequency", "nonlinear")):
    """Compute HRV indices across the requested domains in one call.

    A thin umbrella over [`hrv_time`][physiotrack.signals.hrv_time],
    [`hrv_frequency`][physiotrack.signals.hrv_frequency] and
    [`hrv_nonlinear`][physiotrack.signals.hrv_nonlinear]; the merged dict is what the
    estimator, overlays and exports consume.

    Args:
        rri_ms (np.ndarray): RR-interval series in milliseconds (e.g. from
            [`bvp_to_rri`][physiotrack.signals.bvp_to_rri]).
        t_sec (np.ndarray, optional): Beat timestamps in seconds for the frequency
            domain. Defaults to ``None`` (cumulative RR times).
        domains (Sequence[str], optional): Any of ``"time"``, ``"frequency"``,
            ``"nonlinear"``. Defaults to all three.

    Returns:
        dict: The merged indices from the selected domains.

    Example:
        ```python
        from physiotrack.signals import bvp_to_rri, compute_hrv
        rri_ms, t_sec = bvp_to_rri(bvp, fps=30.0)
        hrv = compute_hrv(rri_ms, t_sec, domains=("time", "nonlinear"))
        print(hrv["RMSSD"], hrv["SD1"])
        ```

    See Also:
        [`correct_rr_artifacts`][physiotrack.signals.correct_rr_artifacts]: clean the
            RR series first for reliable indices.
    """
    out = {}
    if "time" in domains:
        out.update(hrv_time(rri_ms))
    if "frequency" in domains:
        out.update(hrv_frequency(rri_ms, t_sec))
    if "nonlinear" in domains:
        out.update(hrv_nonlinear(rri_ms))
    return out
