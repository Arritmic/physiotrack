"""HRV indices cross-checked against closed-form references (and NeuroKit2 if present)."""
import numpy as np
import pytest

from physiotrack.signals import hrv_time, hrv_nonlinear, hrv_frequency, sample_entropy
from physiotrack.signals.ppg.hrv import approximate_entropy
from _synth import synth_rri


def test_time_domain_matches_closed_form():
    rri = synth_rri(mean_ms=850.0, n=300, sdnn_ms=45.0)
    idx = hrv_time(rri)
    diff = np.diff(rri)
    assert np.isclose(idx["MeanNN"], rri.mean())
    assert np.isclose(idx["SDNN"], np.std(rri, ddof=1))
    assert np.isclose(idx["RMSSD"], np.sqrt(np.mean(diff ** 2)))
    assert np.isclose(idx["SDSD"], np.std(diff, ddof=1))
    assert np.isclose(idx["pNN50"], 100.0 * np.sum(np.abs(diff) > 50) / rri.size)


def test_poincare_identities():
    rri = synth_rri(n=200)
    nl = hrv_nonlinear(rri)
    sdsd = np.std(np.diff(rri), ddof=1)
    sdnn = np.std(rri, ddof=1)
    assert np.isclose(nl["SD1"], np.sqrt(0.5) * sdsd)
    assert np.isclose(nl["SD2"], np.sqrt(max(2 * sdnn ** 2 - 0.5 * sdsd ** 2, 0.0)))
    assert np.isclose(nl["S"], np.pi * nl["SD1"] * nl["SD2"])


def test_sample_entropy_brute_force_reference():
    x = np.random.RandomState(0).randn(120)
    m, r = 2, 0.2 * np.std(x, ddof=1)

    def brute(k):
        vecs = np.array([x[i:i + k] for i in range(len(x) - k + 1)])
        c = 0
        for i in range(len(vecs)):
            d = np.max(np.abs(vecs - vecs[i]), axis=1)
            c += int(np.sum(d <= r)) - 1
        return c

    expected = -np.log(brute(m + 1) / brute(m))
    assert np.isclose(sample_entropy(x, m=m), expected)


def test_frequency_domain_locates_known_hf_tone():
    # RR series oscillating at 0.25 Hz in *real time* (HF band = 0.15-0.40 Hz).
    # Beat i occurs at ~0.8 s * i, so build the modulation on that time axis.
    n = 400
    approx_t = 0.8 * np.arange(n)          # cumulative beat time (s)
    rri = 800.0 + 30.0 * np.sin(2 * np.pi * 0.25 * approx_t)
    f = hrv_frequency(rri, t_sec=np.cumsum(rri) / 1000.0)
    assert f["HF"] > f["LF"]               # dominant power falls in the HF band


def test_short_series_returns_nan():
    assert np.isnan(hrv_time(np.array([800.0]))["SDNN"])


def test_cross_check_neurokit2_if_available():
    nk = pytest.importorskip("neurokit2")
    rri = synth_rri(mean_ms=800.0, n=300, sdnn_ms=40.0)
    ours = hrv_time(rri)
    peaks = np.cumsum(rri)  # ms, 1000 Hz grid
    nk_idx = nk.hrv_time(peaks=peaks.astype(int), sampling_rate=1000, show=False)
    assert abs(ours["RMSSD"] - float(nk_idx["HRV_RMSSD"].iloc[0])) < 1.0
    assert abs(ours["SDNN"] - float(nk_idx["HRV_SDNN"].iloc[0])) < 1.0
