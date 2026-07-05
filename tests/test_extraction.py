"""rPPG extraction methods: HR recovery + regression for the POS/CHROM fixes."""
import numpy as np
import pytest

from physiotrack.signals import POS, CHROM, LGI, OMIT, bandpass_filter, bvp_to_hr
from _synth import synth_rgb


def _recovered_bpm(bvp, fps):
    bvp = bandpass_filter(bvp, 0.7, 4.0, fps)
    hr, _ = bvp_to_hr(bvp, fps, lo_hz=0.7, hi_hz=4.0)
    return float(hr[-1]) if hr.size else np.nan


@pytest.mark.parametrize("method", [POS, CHROM, LGI, OMIT])
def test_method_recovers_known_hr(method):
    rgb, _ = synth_rgb(hr_bpm=72.0, fps=30.0, seconds=25.0)
    bvp = method(30.0).apply(rgb)
    assert bvp.shape[0] == rgb.shape[1]
    assert np.all(np.isfinite(bvp))
    assert abs(_recovered_bpm(bvp, 30.0) - 72.0) < 5.0


def test_pos_short_window_is_not_all_zeros():
    # Trace shorter than the 1.6 s window must still produce a pulse (regression).
    rgb, _ = synth_rgb(fps=30.0, seconds=1.0)   # 30 samples < 48 (=1.6*30)
    bvp = POS(30.0).apply(rgb)
    assert np.any(bvp != 0.0)


def test_chrom_matches_dehaan_windowed_reference():
    # CHROM must track the canonical de Haan & Jeanne (2013) windowed reference.
    fps = 30.0
    rgb, _ = synth_rgb(hr_bpm=66.0, fps=fps, seconds=25.0)

    from scipy.signal import butter, filtfilt
    def bp(x):
        b, a = butter(4, [0.5, 4.0], btype="bandpass", fs=fps)
        return filtfilt(b, a, x)
    # Filter both with the same zero-phase filter so waveforms are comparable.
    ours = bp(CHROM(fps).apply(rgb))
    R, G, B = rgb
    N = rgb.shape[1]
    l = int(1.6 * fps)
    ref = np.zeros(N)
    win = np.hanning(l)
    for m in range(0, N - l + 1):
        Rn, Gn, Bn = (R[m:m+l] / R[m:m+l].mean(),
                      G[m:m+l] / G[m:m+l].mean(),
                      B[m:m+l] / B[m:m+l].mean())
        Xf, Yf = bp(3*Rn - 2*Gn), bp(1.5*Rn + Gn - 1.5*Bn)
        S = Xf - (Xf.std() / Yf.std()) * Yf
        ref[m:m+l] += win * (S - S.mean())
    ref = bp(ref)
    corr = abs(np.corrcoef(ours / ours.std(), ref / ref.std())[0, 1])
    assert corr > 0.9
