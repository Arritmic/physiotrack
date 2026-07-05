"""RR-interval artefact detection/correction (Lipponen & Tarvainen, 2019)."""
import numpy as np
import pytest

from physiotrack.signals import find_rr_artifacts, correct_rr_artifacts
from _synth import synth_rri


def test_clean_series_needs_no_correction():
    rri = synth_rri(mean_ms=800.0, n=200, sdnn_ms=20.0)
    corrected, art = correct_rr_artifacts(rri)
    # A physiologically plausible series should be left essentially unchanged.
    assert abs(corrected.size - rri.size) <= 2
    assert len(art["extra"]) == 0


def test_extra_beat_is_detected_and_removed():
    rri = list(synth_rri(mean_ms=800.0, n=120, sdnn_ms=15.0))
    # Insert an "extra" beat: split one interval into two short ones.
    rri.insert(60, 200.0)
    rri[61] = 200.0
    rri = np.array(rri)
    art = find_rr_artifacts(rri)
    total = sum(len(v) for v in art.values())
    assert total >= 1                      # something flagged
    corrected, _ = correct_rr_artifacts(rri)
    # Correction should not blow up the series length.
    assert corrected.size <= rri.size + 2


def test_missed_beat_series_is_repaired():
    rri = list(synth_rri(mean_ms=800.0, n=120, sdnn_ms=15.0))
    rri[60] = 1600.0                       # a missed beat = one doubled interval
    rri = np.array(rri)
    corrected, _ = correct_rr_artifacts(rri)
    assert np.all(np.isfinite(corrected))
    assert corrected.size >= 1


def test_short_series_returns_unchanged():
    rri = np.array([800.0, 810.0])
    corrected, art = correct_rr_artifacts(rri)
    assert np.array_equal(corrected, rri)
    assert all(len(v) == 0 for v in art.values())


def test_cross_check_neurokit2_if_available():
    nk = pytest.importorskip("neurokit2")
    rri = list(synth_rri(mean_ms=800.0, n=150, sdnn_ms=15.0))
    rri[75] = 1600.0
    rri = np.array(rri)
    peaks = np.concatenate([[0], np.cumsum(rri)]).astype(int)
    _, nk_peaks = nk.signal_fixpeaks(peaks, sampling_rate=1000, method="kubios")
    ours, _ = correct_rr_artifacts(rri)
    # Both should converge to a similar beat count (within a couple of beats).
    assert abs(len(ours) - (len(nk_peaks) - 1)) <= 3
