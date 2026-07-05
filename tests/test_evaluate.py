"""Signal-agreement metrics: correctness of the fixed phase/event functions."""
import numpy as np

from physiotrack.signals import phase_synchrony, event_synchronization


def test_phase_synchrony_in_phase_near_one_across_wrap():
    # Two nearly in-phase signals whose instantaneous phase straddles the +/-pi
    # wrap must score near 1, not near -1 (the pre-fix bug).
    t = np.arange(0, 10, 1 / 100.0)
    a = np.sin(2 * np.pi * 1.0 * t)
    b = np.sin(2 * np.pi * 1.0 * t + 0.01)
    ps = phase_synchrony(a, b)
    assert 0.9 <= ps <= 1.0


def test_phase_synchrony_bounded_below_by_zero():
    t = np.arange(0, 10, 1 / 100.0)
    a = np.sin(2 * np.pi * 1.0 * t)
    b = np.sin(2 * np.pi * 1.3 * t)     # different frequency -> low synchrony
    ps = phase_synchrony(a, b)
    assert -0.05 <= ps <= 1.0           # must never go strongly negative


def test_event_synchronization_no_peaks_returns_zero():
    # Flat signals produce zero peaks; must return 0.0, not ZeroDivisionError.
    assert event_synchronization(np.zeros(200), np.zeros(200)) == 0.0


def test_event_synchronization_identical_signals_is_one():
    t = np.arange(0, 20, 1 / 50.0)
    s = np.sin(2 * np.pi * 0.5 * t)
    assert event_synchronization(s, s) == 1.0
