"""Overlay for the respiration-rate estimate.

A thin *visualization* wrapper over a
:class:`~physiotrack.signals.ppg.estimator.HeartRateEstimator`: the rPPG -> respiration
logic lives in :mod:`~physiotrack.signals.ppg.respiration`; this panel renders the
respiration rate (breaths/min) and the respiratory modulation waveform. Respiration
varies slowly, so it is recomputed at most once every ``recompute_every`` frames.

Share one estimator with the HR / rPPG / HRV panels so rPPG is computed only once::

    est = HeartRateEstimator(method="POS", fps=30.0)
    rsp = RespirationPlotter(estimator=est)
"""

import numpy as np

from physiotrack.signals.plotting._estimator_panel import EstimatorPanel

_ACCENT = (235, 180, 90)   # blue-ish respiration trace (BGR)


class RespirationPlotter(EstimatorPanel):
    """Render the respiration rate + waveform as an on-frame panel.

    Feed frames (and an optional skin/ROI mask) via :meth:`update`, then composite with
    :meth:`attach_to_frame`. Inherits the estimator data API and compositing from
    [`EstimatorPanel`][physiotrack.signals.plotting._estimator_panel.EstimatorPanel];
    pass an existing estimator to share one rPPG computation with the other panels.

    Args:
        *args (Any): Positional args forwarded to ``EstimatorPanel`` (``method``, ``fps``).
        source (str, optional): Respiration source. ``"pulse"`` (rPPG amplitude
            modulation) or ``"rsa"`` (RR-interval variation) are pulse-derived and
            recomputed by :meth:`refresh` from the shared estimator. ``"motion"`` is
            pose-derived (shoulder/torso motion): :meth:`refresh` is then a no-op and the
            caller sets :attr:`resp_wave` / :attr:`rate` directly (e.g. the ``Video``
            pipeline feeds it
            [`respiration_from_motion`][physiotrack.signals.respiration_from_motion]).
            Defaults to ``"pulse"``.
        recompute_every (int, optional): Recompute the rate at most once every this many
            frames. Defaults to ``30``.
        **kwargs (Any): Keyword args forwarded to ``EstimatorPanel``.

    Attributes:
        rate (float): Most recent respiration rate in breaths/min (``np.nan`` until ready).
        resp_wave (numpy.ndarray): Most recent band-passed respiratory waveform.

    Example:
        ```python
        import cv2
        from physiotrack.signals import RespirationPlotter

        rsp = RespirationPlotter(method="POS", fps=30.0)
        for frame, skin_mask in stream:
            rsp.update(frame, roi_mask=skin_mask)
            frame = rsp.attach_to_frame(frame, position="top_right")
            cv2.imshow("resp", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ```

    See Also:
        [`respiration_from_pulse`][physiotrack.signals.respiration_from_pulse]: the
            underlying estimator.
    """

    def __init__(self, *args, source: str = "pulse", recompute_every: int = 30,
                 canvas_width: int = 460, canvas_height: int = 170, **kwargs):
        super().__init__(*args, canvas_width=canvas_width, canvas_height=canvas_height,
                         **kwargs)
        self.source = source
        self.recompute_every = max(1, int(recompute_every))
        self.rate = np.nan
        self.resp_wave = np.array([])
        self._frames = 0

    def update(self, frame_bgr, box=None, roi_mask=None, frame_time=None):
        """Ingest one frame and refresh the cached respiration rate on the cadence.

        Args:
            frame_bgr (np.ndarray): BGR frame.
            box (Sequence[float], optional): Face box; used when ``roi_mask`` is None.
            roi_mask (np.ndarray, optional): Boolean skin/ROI mask (takes precedence).
            frame_time (float, optional): Frame timestamp in seconds.

        Returns:
            float | None: The current HR estimate in bpm (from the shared estimator).
        """
        hr = self.estimator.update(frame_bgr, box, roi_mask, frame_time)
        self.refresh()
        return hr

    def refresh(self):
        """Recompute the cached respiration rate on the recompute cadence, without
        touching the estimator window. Use this when the shared estimator has already
        been updated elsewhere (e.g. once per frame in the ``Video`` pipeline)."""
        # Pose-derived respiration is fed in externally (resp_wave / rate set by the
        # caller); there is no shared estimator to read from, so refresh does nothing.
        if self.source == "motion":
            return
        from physiotrack.signals.ppg.respiration import (respiration_from_pulse,
                                                          respiration_from_rri)
        self._frames += 1
        if self._frames % self.recompute_every == 0:
            if self.source == "rsa":
                rri_ms, t_sec = self.estimator.rri()
                self.resp_wave, self.rate = respiration_from_rri(rri_ms, t_sec)
            else:
                self.resp_wave, self.rate = respiration_from_pulse(
                    self.estimator.bvp, self.estimator.fps)

    def render(self):
        """Render a transparent BGRA panel: respiration rate + waveform.

        Returns:
            numpy.ndarray: The panel as a BGRA image of size
                ``(canvas_height, canvas_width, 4)``.
        """
        ov, s, pad = self._new_canvas()
        w, h = self.canvas_width, self.canvas_height
        rate_txt = f"{self.rate:.0f}" if np.isfinite(self.rate) else "--"
        ov.text((pad, int(8 * s)), f"Resp  {rate_txt} br/min", size=24 * s,
                color=(235, 235, 235), bold=True)
        meta = f"{self.source.upper()}"
        mw, _ = ov.measure(meta, 15 * s)
        ov.text((w - mw - pad, int(9 * s)), meta, size=15 * s, color=(180, 180, 190))

        wave = np.asarray(self.resp_wave, dtype=float)
        y0, ph = int(42 * s), h - int(52 * s)
        if wave.size >= 2:
            seg = wave - wave.min()
            rng = seg.max() or 1.0
            xs = np.linspace(pad, w - pad, len(seg))
            pts = [(float(x), float(y0 + ph - (v / rng) * ph)) for x, v in zip(xs, seg)]
            ov.polyline(pts, _ACCENT, width=1.5 * s)
        else:
            ov.text((pad, y0 + ph // 2 - int(9 * s)), "collecting...",
                    size=18 * s, color=(150, 150, 150))
        return ov.render()
