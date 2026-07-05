"""Respiration-rate estimation validated against synthetic modulated signals."""
import numpy as np

from physiotrack.signals import (
    respiration_rate_from_signal, respiration_from_pulse, respiration_from_rri,
)


def test_rate_from_signal_recovers_known_frequency():
    fps, rr_bpm = 30.0, 15.0            # 15 breaths/min = 0.25 Hz
    t = np.arange(0, 60, 1 / fps)
    mod = np.sin(2 * np.pi * (rr_bpm / 60.0) * t)
    rate, _ = respiration_rate_from_signal(mod, fps)
    assert abs(rate - rr_bpm) < 1.5


def test_from_pulse_riav_recovers_breathing_rate():
    # Amplitude-modulate a cardiac pulse at a known breathing rate.
    fps, hr_bpm, rr_bpm = 30.0, 72.0, 12.0
    t = np.arange(0, 60, 1 / fps)
    am = 1.0 + 0.5 * np.sin(2 * np.pi * (rr_bpm / 60.0) * t)
    bvp = am * np.sin(2 * np.pi * (hr_bpm / 60.0) * t)
    _, rate = respiration_from_pulse(bvp, fps, method="riav")
    assert abs(rate - rr_bpm) < 2.0


def test_from_rri_recovers_rsa_frequency():
    # RR series oscillating (RSA) at a known breathing rate.
    rr_bpm = 18.0
    n = 300
    t = np.cumsum(np.full(n, 0.8))       # ~75 bpm baseline
    rri = 800.0 + 40.0 * np.sin(2 * np.pi * (rr_bpm / 60.0) * t)
    _, rate = respiration_from_rri(rri, t)
    assert abs(rate - rr_bpm) < 3.0


def test_short_signal_returns_nan():
    rate, _ = respiration_rate_from_signal(np.ones(4), 30.0)
    assert np.isnan(rate)
