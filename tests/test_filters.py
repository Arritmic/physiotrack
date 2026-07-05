"""Filtering primitives: regression + shape/contract tests."""
import numpy as np

from physiotrack.signals import (
    bandpass_filter, band_pass_filter, bandpass_firwin,
    zero_mean_std_norm, zero_mean_std_norm_1ch,
)


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


def test_band_pass_filter_matches_bandpass_filter():
    # The pair-argument wrapper must equal the single implementation exactly.
    x = np.random.RandomState(0).randn(600)
    a = bandpass_filter(x, 0.7, 4.0, fs=30.0, order=5)
    b = band_pass_filter(x, [0.7, 4.0], fs=30.0, order=5)
    assert np.allclose(a, b)
