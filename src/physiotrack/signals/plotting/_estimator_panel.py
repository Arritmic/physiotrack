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

import numpy as np
from typing import Optional

from physiotrack.core.overlay import OverlayCanvas, alpha_composite
from physiotrack.signals.ppg.estimator import HeartRateEstimator
from ...core.panel import PanelMixin


class EstimatorPanel(PanelMixin):
    """Base panel: wraps a HeartRateEstimator and composites a BGRA panel onto a frame."""

    # Placement and compositing come from PanelMixin; these are this panel's
    # own defaults, preserved exactly as they were before the consolidation.
    PANEL_POSITION = 'bottom_right'
    PANEL_MARGIN = 12
    PANEL_BACKDROP = False


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
        """A high-quality (supersampled, TrueType) overlay panel with the standard
        background + border. Returns ``(OverlayCanvas, s, pad)``; subclasses draw on
        the canvas and return ``canvas.render()``."""
        w, h = self.canvas_width, self.canvas_height
        s = w / 460.0
        ov = OverlayCanvas(w, h, bg=(24, 22, 28), bg_alpha=self.bg_alpha,
                           border=(105, 95, 90), radius=max(2, int(8 * s)))
        return ov, s, max(6, int(10 * s))

    def _draw_bvp(self, ov, color, *, y0, ph, pad, s, window_sec):
        """Draw the band-passed BVP trace, or a 'collecting...' note if too short."""
        bvp = self.estimator.bvp
        if bvp.size >= 2:
            seg = bvp[-int(window_sec * self.estimator.fps):]
            seg = seg - seg.min()
            rng = seg.max() or 1.0
            xs = np.linspace(pad, self.canvas_width - pad, len(seg))
            pts = [(float(x), float(y0 + ph - (v / rng) * ph)) for x, v in zip(xs, seg)]
            ov.polyline(pts, color, width=1.5 * s)
        else:
            ov.text((pad, y0 + ph // 2 - int(9 * s)), "collecting...",
                    size=18 * s, color=(150, 150, 150))

    def render(self) -> np.ndarray:
        """Render the panel as a transparent BGRA image. Implemented by subclasses."""
        raise NotImplementedError
