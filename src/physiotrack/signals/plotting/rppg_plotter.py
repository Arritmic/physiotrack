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

from physiotrack.signals.plotting._estimator_panel import EstimatorPanel

_ACCENT = (120, 220, 120)   # green BVP trace (BGR)


class RPPGPlotter(EstimatorPanel):
    """Render the rPPG blood-volume-pulse (BVP) trace as an on-frame panel.

    A thin visualization over a
    [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator]: feed frames (and an
    optional skin/ROI mask) via :meth:`update`, then composite the pulse-trace panel
    with :meth:`attach_to_frame`. Because the heart rate is derived from this same
    signal, share one estimator with a
    [`HeartRatePlotter`][physiotrack.signals.HeartRatePlotter] to compute rPPG once and
    show both the pulse and the bpm. Inherits the estimator data API and
    ``attach_to_frame`` from ``EstimatorPanel``.

    Attributes:
        display_window_sec (float): Seconds of pulse shown in the trace.
        bvp (numpy.ndarray): Current band-passed BVP buffer.
        method_name (str): Name of the active rPPG method.

    Example:
        ```python
        import cv2
        from physiotrack.signals import RPPGPlotter, HeartRatePlotter, HeartRateEstimator

        est = HeartRateEstimator(method="POS", fps=30.0)
        sig = RPPGPlotter(estimator=est)         # rPPG/BVP pulse
        hrp = HeartRatePlotter(estimator=est)    # derived HR (bpm)
        for frame, skin_mask in stream:
            est.update(frame, roi_mask=skin_mask)     # compute rPPG once
            frame = sig.attach_to_frame(frame, position="top_right")
            frame = hrp.attach_to_frame(frame, position="top_right",
                                        above_element_height=sig.canvas_height)
            cv2.imshow("rppg", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ```

    See Also:
        [`HeartRatePlotter`][physiotrack.signals.HeartRatePlotter]: the derived bpm panel.
    """

    def __init__(self, *args, window_sec: float = 8.0, **kwargs):
        """Create the pulse-trace panel.

        Args:
            *args (Any): Positional args forwarded to ``EstimatorPanel`` (``method``, ``fps``).
            window_sec (float, optional): Seconds of the BVP pulse shown in the trace.
                Defaults to ``8.0``.
            **kwargs (Any): Keyword args forwarded to ``EstimatorPanel`` (``estimator``,
                ``canvas_width``, ``canvas_height``, ``bg_alpha`` and any
                ``HeartRateEstimator`` kwargs).
        """
        super().__init__(*args, **kwargs)
        self.display_window_sec = float(window_sec)   # seconds of pulse shown in the trace

    def render(self):
        """Render a transparent BGRA panel: the rPPG (BVP) pulse trace.

        Returns:
            numpy.ndarray: The panel as a BGRA (4-channel) image of size
                ``(canvas_height, canvas_width, 4)``.
        """
        ov, s, pad = self._new_canvas()
        h = self.canvas_height
        ov.text((pad, int(8 * s)), f"rPPG signal  {self.estimator.method_name}",
                size=22 * s, color=(235, 235, 235), bold=True)
        self._draw_bvp(ov, _ACCENT, y0=int(40 * s), ph=h - int(50 * s),
                       pad=pad, s=s, window_sec=self.display_window_sec)
        return ov.render()
