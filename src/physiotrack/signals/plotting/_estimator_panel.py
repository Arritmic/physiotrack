"""Shared base for rPPG overlay panels.

Both :class:`~physiotrack.signals.plotting.hr_plotter.HeartRatePlotter` and
:class:`~physiotrack.signals.plotting.rppg_plotter.RPPGPlotter` are thin
visualizations of a single :class:`~physiotrack.signals.ppg.estimator.HeartRateEstimator`
(the heart rate is derived from the same band-passed BVP the rPPG panel draws), so
the estimator wiring, the data-API delegation, the BVP-trace drawing and the frame
compositing live here once. Subclasses only implement :meth:`render`.

Pair them on one shared estimator to show the pulse signal and the derived HR
without recomputing rPPG twice::

    est = HeartRateEstimator(method="POS", fps=30.0)
    sig = RPPGPlotter(estimator=est)
    hrp = HeartRatePlotter(estimator=est)
"""

import cv2
import numpy as np
from typing import Optional

from physiotrack.signals.ppg.estimator import HeartRateEstimator


class EstimatorPanel:
    """Base panel: wraps a HeartRateEstimator and composites a BGRA panel onto a frame."""

    def __init__(self,
                 method: str = "POS",
                 fps: float = 30.0,
                 *,
                 estimator: Optional[HeartRateEstimator] = None,
                 canvas_width: int = 460,
                 canvas_height: int = 170,
                 bg_alpha: float = 0.55,
                 **estimator_kwargs):
        # Wrap an existing estimator (share it across panels), or build one.
        self.estimator = estimator if estimator is not None else HeartRateEstimator(
            method, fps, **estimator_kwargs)
        self.canvas_width = int(canvas_width)
        self.canvas_height = int(canvas_height)
        self.bg_alpha = float(np.clip(bg_alpha, 0.0, 1.0))

    # -- delegate the data API to the estimator (so a panel is a drop-in for it) --
    @property
    def hr(self):
        return self.estimator.hr

    @property
    def snr(self):
        return self.estimator.snr

    @property
    def bvp(self):
        return self.estimator.bvp

    @property
    def method_name(self):
        return self.estimator.method_name

    def update(self, frame_bgr, box=None, roi_mask=None, frame_time: Optional[float] = None):
        return self.estimator.update(frame_bgr, box, roi_mask, frame_time)

    def push_rgb(self, r, g, b):
        return self.estimator.push_rgb(r, g, b)

    def clear(self):
        self.estimator.clear()

    # -- shared drawing helpers (``s`` scales everything to the 460 px reference) --
    def _new_canvas(self):
        """Transparent BGRA panel with the standard background + border. Returns (canvas, s, pad)."""
        w, h = self.canvas_width, self.canvas_height
        s = w / 460.0
        canvas = np.zeros((h, w, 4), np.uint8)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (24, 22, 28, int(self.bg_alpha * 255)), -1)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (105, 95, 90, 220), max(1, round(s)))
        return canvas, s, max(6, int(10 * s))

    def _draw_bvp(self, canvas, color, *, y0, ph, pad, s, window_sec):
        """Draw the band-passed BVP trace, or a 'collecting...' note if too short."""
        bvp = self.estimator.bvp
        if bvp.size >= 2:
            seg = bvp[-int(window_sec * self.estimator.fps):]
            seg = seg - seg.min()
            rng = seg.max() or 1.0
            xs = np.linspace(pad, canvas.shape[1] - pad, len(seg))
            pts = [[int(x), int(y0 + ph - (v / rng) * ph)] for x, v in zip(xs, seg)]
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (*color, 255),
                          max(1, round(1.5 * s)), cv2.LINE_AA)
        else:
            cv2.putText(canvas, "collecting...", (pad, y0 + ph // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * s, (150, 150, 150, 255), max(1, round(s)), cv2.LINE_AA)

    def render(self) -> np.ndarray:
        """Render the panel as a transparent BGRA image. Implemented by subclasses."""
        raise NotImplementedError

    def attach_to_frame(self, frame: np.ndarray, position: str = "bottom_right",
                        margin: int = 12, above_element_height: int = 0) -> np.ndarray:
        """Alpha-composite this panel onto a frame corner."""
        canvas = self.render()
        h, w = frame.shape[:2]
        ch, cw = canvas.shape[:2]
        if cw > w - 2 * margin or ch > h - 2 * margin:
            return frame
        extra = margin if above_element_height > 0 else 0
        y1 = (margin + above_element_height + extra) if "top" in position \
            else (h - ch - margin - above_element_height - extra)
        x1 = margin if "left" in position else w - cw - margin
        y1 = int(np.clip(y1, 0, h - ch))
        x1 = int(np.clip(x1, 0, w - cw))
        roi = frame[y1:y1 + ch, x1:x1 + cw].astype(np.float32)
        alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
        blended = alpha * canvas[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi
        out = frame.copy()
        out[y1:y1 + ch, x1:x1 + cw] = blended.astype(np.uint8)
        return out
