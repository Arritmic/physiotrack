"""Real-time remote-PPG heart-rate estimator (no visualization).

The core rPPG -> HR flow, decoupled from any plotting: from a per-frame face box
it segments skin, averages the RGB, accumulates a sliding trace, converts it to a
blood-volume pulse (POS/CHROM/LGI/OMIT), band-passes it, and estimates the heart
rate from the Welch-PSD peak plus a de~Haan SNR. Expose the current values via
``.hr``, ``.snr``, ``.bvp`` or :meth:`values` -- use this directly when you only
need numbers; :class:`~physiotrack.signals.HeartRatePlotter` wraps it for an
overlay. All frequency bands are configurable.

Example (values only)::

    est = HeartRateEstimator(method="POS", fps=30.0, hr_band=(0.75, 4.0))
    for frame, box in stream:
        est.update(frame, box)          # box = face (x1,y1,x2,y2) or None
        print(est.hr, est.snr)

If you already have a skin RGB sample, feed it directly with ``push_rgb(r, g, b)``.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple

from physiotrack.signals.ppg import POS, CHROM, LGI, OMIT
from physiotrack.signals.filters import bandpass_filter
from physiotrack.signals.ppg.metrics import bvp_to_hr, bvp_snr

_METHODS = {"POS": POS, "CHROM": CHROM, "LGI": LGI, "OMIT": OMIT}


class HeartRateEstimator:
    """Sliding-window rPPG heart-rate estimator. Pure data; no rendering."""

    def __init__(self,
                 method: str = "POS",
                 fps: float = 30.0,
                 *,
                 window_sec: float = 10.0,
                 hr_band: Tuple[float, float] = (0.75, 4.0),
                 snr_half_bw_hz: float = 0.1,
                 use_skin_mask: bool = True,
                 min_fill: float = 0.6,
                 smooth_hr: bool = True):
        method = method.upper()
        if method not in _METHODS:
            raise ValueError(f"Unknown rPPG method {method!r}. Choose from {list(_METHODS)}.")
        self.method_name = method
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.method = _METHODS[method](self.fps)
        self.window_sec = float(window_sec)
        self.hr_band = (float(hr_band[0]), float(hr_band[1]))
        self.snr_half_bw_hz = float(snr_half_bw_hz)
        self.use_skin_mask = bool(use_skin_mask)
        self.min_fill = float(min_fill)
        self.smooth_hr = bool(smooth_hr)

        self._win = max(2, int(self.window_sec * self.fps))
        self._rgb = deque(maxlen=self._win)     # appended as (R, G, B)
        self._hr_hist = deque(maxlen=10)
        self.bvp = np.array([])                 # latest band-passed BVP
        self.hr: Optional[float] = None         # latest HR (bpm)
        self.snr: Optional[float] = None        # latest de Haan SNR (dB)

    # ------------------------------------------------------------------ inputs
    # Forehead + cheek sub-regions as fractions (fx1, fy1, fx2, fy2) of the face
    # box -- the high-perfusion, low-motion areas preferred for rPPG. The eyes,
    # nose, mouth, hair and box background are deliberately excluded.
    _ROI_REGIONS = (
        (0.25, 0.10, 0.75, 0.28),   # forehead
        (0.13, 0.52, 0.40, 0.80),   # left cheek
        (0.60, 0.52, 0.87, 0.80),   # right cheek
    )

    @classmethod
    def skin_mean_rgb(cls, frame_bgr, box, use_skin_mask: bool = True):
        """Mean (R, G, B) over the forehead + cheeks of a face box (skin-gated).

        Samples only the high-signal facial regions, not the whole box, and (by
        default) keeps just the skin pixels via a YCrCb mask.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            return None
        pix = []
        for fx1, fy1, fx2, fy2 in cls._ROI_REGIONS:
            rx1, ry1 = x1 + int(fx1 * bw), y1 + int(fy1 * bh)
            rx2, ry2 = x1 + int(fx2 * bw), y1 + int(fy2 * bh)
            sub = frame_bgr[ry1:ry2, rx1:rx2]
            if sub.size == 0:
                continue
            if use_skin_mask:
                ycrcb = cv2.cvtColor(sub, cv2.COLOR_BGR2YCrCb)
                cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
                m = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
                sub = sub[m] if m.any() else sub.reshape(-1, 3)
            else:
                sub = sub.reshape(-1, 3)
            pix.append(sub)
        if not pix:
            return None
        allpx = np.concatenate(pix, axis=0)
        if allpx.shape[0] < 20:
            return None
        return (allpx[:, 2].mean(), allpx[:, 1].mean(), allpx[:, 0].mean())

    @staticmethod
    def mask_mean_rgb(frame_bgr, roi_mask):
        """Mean (R, G, B) over a boolean ROI mask (e.g. a SegFace skin mask)."""
        m = np.asarray(roi_mask, dtype=bool)
        if m.shape[:2] != frame_bgr.shape[:2] or m.sum() < 20:
            return None
        px = frame_bgr[m]
        return (px[:, 2].mean(), px[:, 1].mean(), px[:, 0].mean())

    def push_rgb(self, r: float, g: float, b: float) -> Optional[float]:
        """Feed one skin RGB sample directly (when you do your own ROI)."""
        self._rgb.append((r, g, b))
        if len(self._rgb) >= max(2, int(self.min_fill * self._win)):
            self._recompute()
        return self.hr

    def update(self, frame_bgr, box=None, roi_mask=None,
               frame_time: Optional[float] = None) -> Optional[float]:
        """Ingest one frame; recompute HR once the window is filled.

        Provide a ``roi_mask`` (boolean, frame-sized -- e.g. a SegFace skin mask)
        for segmentation-based sampling, or a face ``box`` to use the built-in
        forehead/cheek skin ROI. ``roi_mask`` takes precedence.
        """
        rgb = None
        if roi_mask is not None:
            rgb = self.mask_mean_rgb(frame_bgr, roi_mask)
        elif box is not None:
            rgb = self.skin_mean_rgb(frame_bgr, box, self.use_skin_mask)
        if rgb is not None:
            self._rgb.append(rgb)
        if len(self._rgb) >= max(2, int(self.min_fill * self._win)):
            self._recompute()
        return self.hr

    # ------------------------------------------------------------------ compute
    def _recompute(self) -> None:
        trace = np.asarray(self._rgb, dtype=float).T          # (3, N) rows R, G, B
        bvp = np.asarray(self.method.apply(trace), dtype=float).ravel()
        if bvp.size < 8:
            return
        try:
            bvp = bandpass_filter(bvp, self.hr_band[0], self.hr_band[1], self.fps)
        except Exception:
            pass
        self.bvp = bvp
        hr, _ = bvp_to_hr(bvp, self.fps, win_sec=self.window_sec, step_sec=self.window_sec,
                          lo_hz=self.hr_band[0], hi_hz=self.hr_band[1])
        if hr.size:
            self._hr_hist.append(float(hr[-1]))
            self.hr = float(np.median(self._hr_hist)) if self.smooth_hr else float(hr[-1])
            self.snr = bvp_snr(bvp, self.fps, self.hr, hi_hz=self.hr_band[1],
                               half_bw_hz=self.snr_half_bw_hz)

    # ------------------------------------------------------------------ access
    def values(self) -> dict:
        """Current estimates: ``{"hr": bpm, "snr": dB, "method": name}``."""
        return {"hr": self.hr, "snr": self.snr, "method": self.method_name}

    def clear(self) -> None:
        self._rgb.clear()
        self._hr_hist.clear()
        self.bvp = np.array([])
        self.hr = self.snr = None
