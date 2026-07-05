"""HR/rPPG metrics: SNR reference-HR behaviour and benchmarking."""
import numpy as np

from physiotrack.signals import bvp_to_hr, bvp_snr, benchmark_rppg_methods
from _synth import synth_bvp, synth_rgb


def test_bvp_to_hr_recovers_known_rate():
    bvp, _ = synth_bvp(hr_bpm=72.0, fps=30.0, seconds=20.0, noise=0.05)
    hr, _ = bvp_to_hr(bvp, 30.0, lo_hz=0.7, hi_hz=4.0)
    assert abs(hr[-1] - 72.0) < 5.0


def test_snr_uses_reference_hr_in_benchmark():
    # With a wrong reference HR, SNR (measured about that reference) must drop
    # relative to the correct reference -- proving it is not self-referential.
    rgb, _ = synth_rgb(hr_bpm=72.0, fps=30.0, seconds=25.0)
    good = benchmark_rppg_methods(rgb, 30.0, ref_hr_bpm=72.0)
    bad = benchmark_rppg_methods(rgb, 30.0, ref_hr_bpm=150.0)
    assert good["POS"]["snr"] > bad["POS"]["snr"]


def test_snr_nan_reference_is_nan():
    bvp, _ = synth_bvp()
    assert np.isnan(bvp_snr(bvp, 30.0, np.nan))
