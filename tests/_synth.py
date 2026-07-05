"""Synthetic-signal generators shared across the signals test suite."""
import numpy as np


def synth_bvp(hr_bpm=72.0, fps=30.0, seconds=20.0, noise=0.02, seed=0):
    """A clean sinusoidal blood-volume pulse at a known heart rate."""
    rs = np.random.RandomState(seed)
    t = np.arange(0, seconds, 1.0 / fps)
    bvp = np.sin(2 * np.pi * (hr_bpm / 60.0) * t) + noise * rs.randn(t.size)
    return bvp, t


def synth_rgb(hr_bpm=72.0, fps=30.0, seconds=20.0, seed=0):
    """A synthetic RGB skin trace (rows R, G, B) with a pulse + skin-tone DC."""
    rs = np.random.RandomState(seed)
    t = np.arange(0, seconds, 1.0 / fps)
    pulse = np.sin(2 * np.pi * (hr_bpm / 60.0) * t)
    dc = np.array([180.0, 120.0, 90.0])
    gain = np.array([0.5, 1.0, 0.3])
    intensity = 0.15 * np.sin(2 * np.pi * 0.25 * t)
    rgb = np.vstack([dc[i] * (1.0 + intensity) + gain[i] * pulse + 0.3 * rs.randn(t.size)
                     for i in range(3)])
    return rgb, t


def synth_rri(mean_ms=800.0, n=256, sdnn_ms=40.0, seed=0):
    """A synthetic RR-interval series (ms) around a mean with controlled variability."""
    rs = np.random.RandomState(seed)
    return mean_ms + sdnn_ms * rs.randn(n)
