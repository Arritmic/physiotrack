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

import numpy as np

from physiotrack.signals.plotting._estimator_panel import EstimatorPanel

_ACCENT = (90, 90, 235)   # red-ish BVP trace (BGR)


class HeartRatePlotter(EstimatorPanel):
    """Render a rPPG heart-rate estimate as an on-frame panel (BVP waveform + bpm).

    A thin visualization over a
    [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator]: feed frames (and an
    optional skin/ROI mask) via :meth:`update`, then composite the panel with
    :meth:`attach_to_frame`. Pass an existing estimator to share one rPPG computation
    with an [`RPPGPlotter`][physiotrack.signals.RPPGPlotter]. Inherits the estimator
    data API (``hr``, ``snr``, ``bvp``, ``method_name``, ``update``, ``push_rgb``,
    ``clear``) and ``attach_to_frame`` from ``EstimatorPanel``.

    Args:
        method (str, optional): rPPG algorithm when building a new estimator, one of
            ``"POS"``, ``"CHROM"``, ``"LGI"``, ``"OMIT"``. Defaults to ``"POS"``.
            Ignored if ``estimator`` is given.
        fps (float, optional): Frame rate for a new estimator. Defaults to ``30.0``.
        estimator (HeartRateEstimator, optional): Existing estimator to wrap/share.
            Defaults to ``None`` (a new one is built from ``method``/``fps``).
        canvas_width (int, optional): Panel width in pixels. Defaults to ``460``.
        canvas_height (int, optional): Panel height in pixels. Defaults to ``170``.
        bg_alpha (float, optional): Panel background opacity in ``[0, 1]``. Defaults to ``0.55``.
        **estimator_kwargs (Any): Forwarded to
            [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator] when one is created.

    Attributes:
        hr (float | None): Current heart rate in bpm.
        snr (float | None): Current signal-to-noise ratio in dB.
        bvp (numpy.ndarray): Current band-passed blood-volume-pulse buffer.
        method_name (str): Name of the active rPPG method.

    Example:
        ```python
        import cv2
        from physiotrack.signals import HeartRatePlotter

        hrp = HeartRatePlotter(method="POS", fps=30.0)
        for frame, skin_mask in stream:
            hrp.update(frame, roi_mask=skin_mask)
            frame = hrp.attach_to_frame(frame, position="bottom_right")
            cv2.imshow("hr", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ```

    See Also:
        [`RPPGPlotter`][physiotrack.signals.RPPGPlotter]: draws the pulse trace it is
            derived from; share one estimator across both.
    """

    def render(self):
        """Render a transparent BGRA panel: BVP waveform + current HR (bpm).

        Fonts/offsets scale with the canvas (``s`` is relative to the 460 px
        reference width) so the panel reads the same at any resolution.

        Returns:
            numpy.ndarray: The panel as a BGRA (4-channel) image of size
                ``(canvas_height, canvas_width, 4)``.
        """
        est = self.estimator
        ov, s, pad = self._new_canvas()
        w, h = self.canvas_width, self.canvas_height

        hr_txt = f"{est.hr:.0f}" if est.hr is not None else "--"
        ov.text((pad, int(8 * s)), f"HR  {hr_txt} bpm", size=26 * s,
                color=(235, 235, 235), bold=True)
        meta = est.method_name
        if est.snr is not None and not np.isnan(est.snr):
            meta += f"   SNR {est.snr:.1f} dB"
        mw, _ = ov.measure(meta, 15 * s)
        ov.text((w - mw - pad, int(7 * s)), meta, size=15 * s, color=(180, 180, 190))

        self._draw_bvp(ov, _ACCENT, y0=int(40 * s), ph=h - int(50 * s),
                       pad=pad, s=s, window_sec=6.0)
        return ov.render()
