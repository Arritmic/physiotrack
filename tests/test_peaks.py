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
