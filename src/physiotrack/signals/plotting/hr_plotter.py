"""Overlay for the real-time rPPG heart-rate estimator.

This is a thin *visualization* wrapper: all the rPPG -> HR logic lives in
:class:`~physiotrack.signals.ppg.estimator.HeartRateEstimator`, and this class only
renders its state (the band-passed BVP waveform + the current HR in bpm) as a panel
that composites onto a frame. The estimator wiring, data-API delegation and
compositing are shared with :class:`~physiotrack.signals.plotting.rppg_plotter.RPPGPlotter`
via :class:`EstimatorPanel`.

Use the estimator directly when you just want values; use this when you want the
overlay. Pass your own estimator to share one across panels::

    est = HeartRateEstimator(method="POS", fps=30.0)
    plot = HeartRatePlotter(estimator=est)
"""

import cv2
import numpy as np

from physiotrack.signals.plotting._estimator_panel import EstimatorPanel

_ACCENT = (90, 90, 235)   # red-ish BVP trace (BGR)


class HeartRatePlotter(EstimatorPanel):
    """Renders a :class:`HeartRateEstimator`'s output as an on-frame panel (BVP + bpm)."""

    def render(self) -> np.ndarray:
        """Render a transparent BGRA panel: BVP waveform + current HR (bpm).

        Fonts/offsets scale with the canvas (``s`` is relative to the 460 px
        reference width) so the panel reads the same at any resolution.
        """
        est = self.estimator
        canvas, s, pad = self._new_canvas()
        w, h = self.canvas_width, self.canvas_height
        th = max(1, round(2 * s))
        font = cv2.FONT_HERSHEY_SIMPLEX

        hr_txt = f"{est.hr:.0f}" if est.hr is not None else "--"
        cv2.putText(canvas, f"HR  {hr_txt} bpm", (pad, int(26 * s)), font, 0.7 * s,
                    (235, 235, 235, 255), th, cv2.LINE_AA)
        meta = est.method_name
        if est.snr is not None and not np.isnan(est.snr):
            meta += f"   SNR {est.snr:.1f} dB"
        (mw, _), _ = cv2.getTextSize(meta, font, 0.42 * s, max(1, round(s)))
        cv2.putText(canvas, meta, (w - mw - pad, int(20 * s)), font, 0.42 * s,
                    (180, 180, 190, 255), max(1, round(s)), cv2.LINE_AA)

        self._draw_bvp(canvas, _ACCENT, y0=int(40 * s), ph=h - int(50 * s),
                       pad=pad, s=s, window_sec=6.0)
        return canvas
