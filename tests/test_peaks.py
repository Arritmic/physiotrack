"""Systolic-peak / RR-interval extraction."""
import numpy as np
import pytest

from physiotrack.signals import detect_pulse_peaks, bvp_to_rri
from _synth import synth_bvp


def test_peak_spacing_matches_hr():
    fps, hr = 30.0, 60.0     # 60 bpm -> one beat per second -> ~fps samples apart
    bvp, _ = synth_bvp(hr_bpm=hr, fps=fps, seconds=20.0, noise=0.0)
    peaks = detect_pulse_peaks(bvp, fps)
    spacing = np.diff(peaks)
    assert abs(np.median(spacing) - fps) <= 2


def test_rri_units_and_alignment():
    fps, hr = 30.0, 75.0
    bvp, _ = synth_bvp(hr_bpm=hr, fps=fps, seconds=20.0, noise=0.0)
    rri_ms, t_sec = bvp_to_rri(bvp, fps)
    assert rri_ms.size == t_sec.size
    # 75 bpm -> ~800 ms RR intervals.
    assert abs(np.median(rri_ms) - 800.0) < 60.0
    # t_sec is the timestamp of the *second* peak of each interval (strictly increasing).
    assert np.all(np.diff(t_sec) > 0)


def test_empty_and_short_inputs():
    assert detect_pulse_peaks(np.array([1.0]), 30.0).size == 0
    rri, t = bvp_to_rri(np.zeros(5), 30.0)
    assert rri.size == 0 and t.size == 0


def test_invalid_fps_raises():
    bvp, _ = synth_bvp()
    with pytest.raises(ValueError):
        detect_pulse_peaks(bvp, 0.0)


def test_nonfinite_bvp_raises():
    bvp, _ = synth_bvp()
    bvp[10] = np.nan
    with pytest.raises(ValueError):
        detect_pulse_peaks(bvp, 30.0)


# --- refractory period: the dicrotic notch must not be counted as a beat -------------
# Regression guard. detect_pulse_peaks used to space beats by the *ceiling* of the
# heart-rate band (fps / 4 Hz = 240 bpm), which admitted the dicrotic notch following
# each systolic peak as a second beat for any rate below ~100 bpm. Since every HRV index
# is computed from these intervals, that halved MeanNN and inflated RMSSD/SDNN/pNN50 into
# physiologically impossible values. Cross-checked against NeuroKit2's ppg_findpeaks,
# which implements Elgendi et al. (2013).

_FPS = 30.0
_DUR = 30.0


def _synthetic_ppg(bpm, notch=0.45, noise=0.05, seed=0):
    """A PPG-like wave with an explicit dicrotic notch at a known rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, _DUR, 1 / _FPS)
    phase = 2 * np.pi * (bpm / 60.0) * t
    return np.sin(phase) + notch * np.sin(2 * phase + 0.9) \
        + noise * rng.standard_normal(t.size)


def _bpm_from_peaks(idx):
    if len(idx) < 2:
        return float("nan")
    return 60000.0 / np.mean(np.diff(idx) / _FPS * 1000.0)


@pytest.mark.parametrize("true_bpm", [50, 60, 72, 78, 90, 100, 120])
def test_detected_rate_matches_the_true_rate(true_bpm):
    peaks = detect_pulse_peaks(_synthetic_ppg(true_bpm), _FPS)
    # Within 2 bpm: the discretisation of peak indices at 30 fps bounds the precision.
    assert _bpm_from_peaks(peaks) == pytest.approx(true_bpm, abs=2.0)


def test_dicrotic_notch_is_not_counted_as_a_beat():
    # A strong notch is the adversarial case: the old band-ceiling rule reported ~118 bpm
    # for this 78 bpm signal.
    peaks = detect_pulse_peaks(_synthetic_ppg(78, notch=0.6), _FPS)
    expected_beats = _DUR * 78 / 60
    assert len(peaks) == pytest.approx(expected_beats, abs=2)


@pytest.mark.parametrize("refractory", [0.5, 0.6, 0.7, 0.8])
def test_result_is_insensitive_to_the_refractory_fraction(refractory):
    peaks = detect_pulse_peaks(_synthetic_ppg(78), _FPS, refractory=refractory)
    assert _bpm_from_peaks(peaks) == pytest.approx(78.0, abs=2.0)


def test_refractory_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="refractory"):
        detect_pulse_peaks(_synthetic_ppg(78), _FPS, refractory=0.0)
    with pytest.raises(ValueError, match="refractory"):
        detect_pulse_peaks(_synthetic_ppg(78), _FPS, refractory=1.5)


@pytest.mark.parametrize("true_bpm", [50, 60, 78, 100, 120])
def test_agrees_with_neurokit2_ppg_findpeaks(true_bpm):
    nk = pytest.importorskip("neurokit2")
    signal = _synthetic_ppg(true_bpm)
    ours = _bpm_from_peaks(detect_pulse_peaks(signal, _FPS))
    theirs = _bpm_from_peaks(
        np.asarray(nk.ppg_findpeaks(signal, sampling_rate=int(_FPS))["PPG_Peaks"]))
    assert ours == pytest.approx(theirs, abs=0.5)


def test_hrv_is_consistent_with_the_spectral_heart_rate():
    """MeanNN must imply the same rate the spectrum reports.

    This is the invariant that exposed the bug: HR came from the Welch peak and HRV from
    beat detection, and the two disagreed by a factor of ~1.8.
    """
    from physiotrack.signals import compute_hrv

    rri_ms, t_sec = bvp_to_rri(_synthetic_ppg(78), _FPS)
    indices = compute_hrv(rri_ms, t_sec, domains=("time",))
    assert 60000.0 / indices["MeanNN"] == pytest.approx(78.0, abs=2.0)
