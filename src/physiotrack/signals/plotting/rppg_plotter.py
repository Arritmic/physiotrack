"""Overlay for the rPPG blood-volume-pulse (BVP) signal.

A thin *visualization* wrapper: the rPPG -> BVP logic lives in
:class:`~physiotrack.signals.ppg.estimator.HeartRateEstimator`; this draws its
band-passed pulse trace as a panel. The estimator wiring, data-API delegation and
compositing are shared with
:class:`~physiotrack.signals.plotting.hr_plotter.HeartRatePlotter` via
:class:`EstimatorPanel`.

Because the heart rate is derived from this same signal, pair the two on one shared
estimator to show the pulse and the HR with a single rPPG computation::

    est = HeartRateEstimator(method="POS", fps=30.0)
    sig = RPPGPlotter(estimator=est)         # rPPG/BVP pulse
    hrp = HeartRatePlotter(estimator=est)    # derived HR (bpm)
    for frame, mask in stream:
        est.update(frame, roi_mask=mask)     # compute once
        frame = sig.attach_to_frame(frame, position="top_right")
        frame = hrp.attach_to_frame(frame, position="top_right",
                                    above_element_height=sig.canvas_height)
"""

import cv2
import numpy as np

from physiotrack.signals.plotting._estimator_panel import EstimatorPanel

_ACCENT = (120, 220, 120)   # green BVP trace (BGR)


class RPPGPlotter(EstimatorPanel):
    """Renders a :class:`HeartRateEstimator`'s band-passed BVP pulse as an on-frame panel."""

    def __init__(self, *args, window_sec: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_window_sec = float(window_sec)   # seconds of pulse shown in the trace

    def render(self) -> np.ndarray:
        """Render a transparent BGRA panel: the rPPG (BVP) pulse trace."""
        canvas, s, pad = self._new_canvas()
        h = self.canvas_height
        cv2.putText(canvas, f"rPPG signal  {self.estimator.method_name}", (pad, int(26 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (235, 235, 235, 255),
                    max(1, round(2 * s)), cv2.LINE_AA)
        self._draw_bvp(canvas, _ACCENT, y0=int(40 * s), ph=h - int(50 * s),
                       pad=pad, s=s, window_sec=self.display_window_sec)
        return canvas
