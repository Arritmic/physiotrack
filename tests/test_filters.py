"""Filtering primitives: regression + shape/contract tests."""
import numpy as np
import pytest
from scipy.signal import find_peaks

from physiotrack.signals import (
    bandpass_filter, bandpass_firwin,
    zero_mean_std_norm, zero_mean_std_norm_1ch,
)


def test_bandpass_filter_is_zero_phase():
    """Peak timing must survive filtering.

    Regression for the forward-only ``lfilter`` implementation, which imposed a
    ~200 ms group delay at 30 fps. That delay propagates into pulse-peak detection,
    the R-R interval series, and therefore every HRV index, so the band-pass must be
    zero phase.
    """
    fs, hr_hz = 30.0, 1.2  # 72 bpm
    t = np.arange(0, 20, 1 / fs)
    clean = np.sin(2 * np.pi * hr_hz * t)
    rng = np.random.default_rng(0)
    noisy = clean + 0.3 * rng.standard_normal(t.size) + 2.0  # noise + DC offset

    def peak_times(x):
        idx, _ = find_peaks(x, distance=int(fs / (hr_hz * 1.6)))
        return t[idx]

    ref = peak_times(clean)
    got = peak_times(bandpass_filter(noisy, 0.75, 4.0, fs))

    n = min(len(ref), len(got))
    assert n >= 10, "expected the pulse peaks to survive filtering"
    shift_s = float(np.mean(got[:n] - ref[:n]))
    # One sample at 30 fps is 33 ms; allow half a sample of discretisation error.
    assert abs(shift_s) < 1.0 / fs, f"phase shift {shift_s * 1000:+.1f} ms is not zero-phase"


def test_bandpass_filter_preserves_shape_2d():
    x = np.random.default_rng(1).standard_normal((2, 600))
    assert bandpass_filter(x, 0.75, 4.0, 30.0).shape == x.shape


def test_bandpass_filter_rejects_too_short_signal():
    # Zero-phase filtering cannot be done on a signal shorter than the backward
    # pass's padding; that must be reported, not silently degraded.
    with pytest.raises(ValueError, match="too short"):
        bandpass_filter(np.zeros(20), 0.75, 4.0, 30.0)


def test_bandpass_firwin_runs_on_modern_scipy():
    # Regression for the removed ``nyq=`` keyword (SciPy >= 1.12): must not raise.
    taps = bandpass_firwin(65, 0.7, 4.0, fs=30.0)
    assert taps.shape == (65,)
    assert np.all(np.isfinite(taps))


def test_zero_mean_std_norm_1ch_preserves_shape():
    x = np.random.RandomState(0).rand(300)
    z = zero_mean_std_norm_1ch(x)
    assert z.shape == x.shape          # was (1, 300) before the fix
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9


def test_zero_mean_std_norm_1ch_constant_input_is_finite():
    z = zero_mean_std_norm_1ch(np.ones(50))
    assert np.all(np.isfinite(z))


def test_zero_mean_std_norm_guards_constant_channel():
    x = np.vstack([np.ones(100), np.random.RandomState(0).rand(100)])
    z = zero_mean_std_norm(x)
    assert np.all(np.isfinite(z))      # constant row must not produce NaN
