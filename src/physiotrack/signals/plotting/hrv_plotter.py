"""Overlay for heart-rate-variability (HRV) indices.

A thin *visualization* wrapper over a
:class:`~physiotrack.signals.ppg.estimator.HeartRateEstimator`: all the rPPG -> RR ->
HRV logic lives in the estimator and
:mod:`~physiotrack.signals.ppg.hrv`; this panel only renders a compact grid of the
key indices (RMSSD, SDNN, pNN50, SD1, SD2, LF/HF). HRV varies slowly, so it is
recomputed at most once every ``recompute_every`` frames rather than per frame.

Share one estimator with the HR / rPPG panels so rPPG is computed only once::

    est = HeartRateEstimator(method="POS", fps=30.0, window_sec=60)
    hrv = HRVPlotter(estimator=est)
"""

import numpy as np

from physiotrack.signals.plotting._estimator_panel import EstimatorPanel


class HRVPlotter(EstimatorPanel):
    """Render heart-rate-variability indices as an on-frame panel.

    Feed frames (and an optional skin/ROI mask) via :meth:`update`, then composite the
    panel with :meth:`attach_to_frame`. Inherits the estimator data API and compositing
    from [`EstimatorPanel`][physiotrack.signals.plotting._estimator_panel.EstimatorPanel];
    pass an existing estimator to share one rPPG computation with the HR / rPPG panels.

    Because HRV needs a longer window than HR, build the shared estimator with a larger
    ``window_sec`` (e.g. 60 s) for stable indices.

    Args:
        *args (Any): Positional args forwarded to ``EstimatorPanel`` (``method``, ``fps``).
        recompute_every (int, optional): Recompute HRV at most once every this many
            frames. Defaults to ``30``.
        correct (bool, optional): Artefact-correct the RR series before computing HRV.
            Defaults to ``True``.
        **kwargs (Any): Keyword args forwarded to ``EstimatorPanel`` (``estimator``,
            ``canvas_width``, ``canvas_height``, ``bg_alpha`` and ``HeartRateEstimator``
            kwargs).

    Attributes:
        hrv (dict): The most recently computed HRV indices (empty until ready).

    Example:
        ```python
        import cv2
        from physiotrack.signals import HRVPlotter

        hrv = HRVPlotter(method="POS", fps=30.0, window_sec=60)
        for frame, skin_mask in stream:
            hrv.update(frame, roi_mask=skin_mask)
            frame = hrv.attach_to_frame(frame, position="top_right")
            cv2.imshow("hrv", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ```

    See Also:
        [`HeartRatePlotter`][physiotrack.signals.HeartRatePlotter],
        [`RPPGPlotter`][physiotrack.signals.RPPGPlotter]: share one estimator across all.
    """

    def __init__(self, *args, recompute_every: int = 30, correct: bool = True,
                 canvas_width: int = 460, canvas_height: int = 190, **kwargs):
        super().__init__(*args, canvas_width=canvas_width, canvas_height=canvas_height,
                         **kwargs)
        self.recompute_every = max(1, int(recompute_every))
        self.correct = bool(correct)
        self.hrv = {}
        self._frames = 0

    def update(self, frame_bgr, box=None, roi_mask=None, frame_time=None):
        """Ingest one frame and refresh the cached HRV on the recompute cadence.

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
        """Recompute the cached HRV on the recompute cadence, without touching the
        estimator window. Use this when the shared estimator has already been updated
        elsewhere (e.g. once per frame in the ``Video`` pipeline) so rPPG is not
        recomputed per panel."""
        self._frames += 1
        if self._frames % self.recompute_every == 0:
            self.hrv = self.estimator.hrv(correct=self.correct)

    def render(self):
        """Render a transparent BGRA panel with the key HRV indices.

        Returns:
            numpy.ndarray: The panel as a BGRA image of size
                ``(canvas_height, canvas_width, 4)``.
        """
        ov, s, pad = self._new_canvas()
        w = self.canvas_width
        ov.text((pad, int(8 * s)), "HRV", size=24 * s, color=(235, 235, 235), bold=True)
        hr = self.estimator.hr
        hr_txt = f"{hr:.0f} bpm" if hr is not None else "-- bpm"
        mw, _ = ov.measure(hr_txt, 16 * s)
        ov.text((w - mw - pad, int(11 * s)), hr_txt, size=16 * s, color=(180, 180, 190))

        if not self.hrv:
            ov.text((pad, int(self.canvas_height / 2)), "collecting...",
                    size=18 * s, color=(150, 150, 150))
            return ov.render()

        def _fmt(key, unit=""):
            v = self.hrv.get(key)
            return f"{v:.1f}{unit}" if isinstance(v, (int, float)) and np.isfinite(v) else "--"

        rows = [
            (f"RMSSD  {_fmt('RMSSD', ' ms')}", f"SD1  {_fmt('SD1', ' ms')}"),
            (f"SDNN   {_fmt('SDNN', ' ms')}", f"SD2  {_fmt('SD2', ' ms')}"),
            (f"pNN50  {_fmt('pNN50', ' %')}", f"LF/HF {_fmt('LFHF')}"),
        ]
        y = int(44 * s)
        for left, right in rows:
            ov.text((pad, y), left, size=17 * s, color=(220, 220, 225))
            ov.text((int(w * 0.52), y), right, size=17 * s, color=(220, 220, 225))
            y += int(46 * s)
        return ov.render()
