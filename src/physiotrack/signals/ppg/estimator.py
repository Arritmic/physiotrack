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

from physiotrack.signals.filters import bandpass_filter
from physiotrack.signals.ppg.metrics import bvp_to_hr, bvp_snr
from physiotrack.signals.ppg.constants import RPPG_METHODS, HR_BAND, DEFAULT_RPPG_METHOD
from physiotrack.signals.ppg.peaks import bvp_to_rri
from physiotrack.signals.ppg.artifacts import correct_rr_artifacts
from physiotrack.signals.ppg.hrv import compute_hrv
from physiotrack.signals.ppg.respiration import respiration_from_pulse, respiration_from_rri


class HeartRateEstimator:
    """Sliding-window rPPG heart-rate estimator (pure data; no rendering).

    Accumulates per-frame skin RGB samples in a sliding window, converts them to a
    blood-volume pulse with one of the rPPG methods (POS/CHROM/LGI/OMIT),
    band-passes it, and estimates the heart rate from the Welch-PSD peak plus a
    de~Haan SNR. Feed frames with :meth:`update` (using a ``roi_mask`` or face
    ``box``) or push RGB directly with :meth:`push_rgb`; read the latest values
    from :attr:`hr`, :attr:`snr`, :attr:`bvp`, or :meth:`values`.

    The same sliding window also yields the downstream pulse analytics without any
    extra buffering: RR-interval (inter-beat) series via :meth:`rri`, heart-rate
    variability via :meth:`hrv`, and respiration rate via :meth:`respiration_rate`.
    HRV needs a longer window than HR -- build the estimator with a larger
    ``window_sec`` (e.g. 60 s) when you need stable HRV indices.

    Attributes:
        method_name (str): Active rPPG method, one of ``"POS"``, ``"CHROM"``,
            ``"LGI"``, ``"OMIT"``.
        fps (float): Sampling rate in Hz used by the method and HR estimation.
        window_sec (float): Sliding-window length in seconds.
        hr_band (tuple[float, float]): ``(lo_hz, hi_hz)`` band-pass / HR-search
            band in Hz.
        snr_half_bw_hz (float): Half-bandwidth in Hz used by the SNR computation.
        use_skin_mask (bool): Whether the box-based ROI gates pixels by a YCrCb
            skin mask.
        min_fill (float): Fraction of the window that must be filled before HR is
            (re)computed.
        smooth_hr (bool): Whether the reported HR is a median over recent windows.
        bvp (np.ndarray): Latest band-passed blood-volume-pulse signal.
        hr (float | None): Latest heart-rate estimate in bpm (``None`` until ready).
        snr (float | None): Latest de~Haan SNR in dB (``None`` until ready).

    Example:
        ```python
        from physiotrack.signals import HeartRateEstimator, FaceSkinExtractor
        fs = FaceSkinExtractor()
        est = HeartRateEstimator("POS", fps=30)
        for frame in frames:
            mask, _ = fs.extract(frame)
            est.update(frame, roi_mask=mask)   # call per frame
        print(est.hr, est.snr)                 # HR (bpm), SNR (dB)
        ```

    See Also:
        [`FaceSkinExtractor`][physiotrack.signals.FaceSkinExtractor]: SegFace skin
            ROI source for :meth:`update`.
        [`bvp_to_hr`][physiotrack.signals.bvp_to_hr],
        [`bvp_snr`][physiotrack.signals.bvp_snr]: the underlying HR/SNR functions.
    """

    def __init__(self,
                 method: str = DEFAULT_RPPG_METHOD,
                 fps: float = 30.0,
                 *,
                 window_sec: float = 10.0,
                 hr_band: Tuple[float, float] = HR_BAND,
                 snr_half_bw_hz: float = 0.1,
                 use_skin_mask: bool = True,
                 min_fill: float = 0.6,
                 smooth_hr: bool = True):
        """Initialize the estimator.

        Args:
            method (str, optional): rPPG extraction method (case-insensitive), one
                of ``"POS"``, ``"CHROM"``, ``"LGI"``, ``"OMIT"``. Defaults to
                ``"POS"``.
            fps (float, optional): Sampling rate in Hz; non-positive values fall
                back to ``30.0``. Defaults to ``30.0``.
            window_sec (float, optional): Sliding-window length in seconds.
                Defaults to ``10.0``.
            hr_band (tuple[float, float], optional): ``(lo_hz, hi_hz)`` band-pass
                and HR-search band in Hz. Defaults to ``(0.75, 4.0)`` (45-240 bpm).
            snr_half_bw_hz (float, optional): Half-bandwidth in Hz around the HR
                fundamental/harmonic for the SNR. Defaults to ``0.1``.
            use_skin_mask (bool, optional): Gate box-based ROI pixels with a YCrCb
                skin mask. Defaults to ``True``.
            min_fill (float, optional): Fraction of the window in ``[0, 1]`` that
                must be filled before HR is recomputed. Defaults to ``0.6``.
            smooth_hr (bool, optional): Report the median HR over the last few
                windows instead of the newest value. Defaults to ``True``.

        Raises:
            ValueError: If ``method`` is not one of the supported names.
        """
        method = method.upper()
        if method not in RPPG_METHODS:
            raise ValueError(f"Unknown rPPG method {method!r}. Choose from {list(RPPG_METHODS)}.")
        self.method_name = method
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.method = RPPG_METHODS[method](self.fps)
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

        Samples only the high-signal facial regions (forehead and both cheeks, per
        :attr:`_ROI_REGIONS`), not the whole box, and (by default) keeps just the
        skin pixels via a YCrCb mask.

        Args:
            frame_bgr (np.ndarray): BGR frame of shape ``(H, W, 3)``.
            box (Sequence[float]): Face box ``(x1, y1, x2, y2)`` in pixels; extra
                trailing values are ignored.
            use_skin_mask (bool, optional): Keep only YCrCb skin pixels within each
                sub-region. Defaults to ``True``.

        Returns:
            tuple[float, float, float] | None: Mean ``(R, G, B)`` over the sampled
                skin pixels, or ``None`` if the box is too small or too few pixels
                are available.
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
        """Mean (R, G, B) over a boolean ROI mask (e.g. a SegFace skin mask).

        Args:
            frame_bgr (np.ndarray): BGR frame of shape ``(H, W, 3)``.
            roi_mask (np.ndarray): Boolean mask of shape ``(H, W)`` selecting the
                pixels to average.

        Returns:
            tuple[float, float, float] | None: Mean ``(R, G, B)`` over the masked
                pixels, or ``None`` if the mask shape mismatches the frame or fewer
                than 20 pixels are selected.
        """
        m = np.asarray(roi_mask, dtype=bool)
        if m.shape[:2] != frame_bgr.shape[:2] or m.sum() < 20:
            return None
        px = frame_bgr[m]
        return (px[:, 2].mean(), px[:, 1].mean(), px[:, 0].mean())

    def push_rgb(self, r: float, g: float, b: float) -> Optional[float]:
        """Feed one skin RGB sample directly (when you do your own ROI).

        Appends the sample to the sliding window and recomputes HR once the window
        is at least :attr:`min_fill` full.

        Args:
            r (float): Mean red channel value for this frame.
            g (float): Mean green channel value for this frame.
            b (float): Mean blue channel value for this frame.

        Returns:
            float | None: The current HR estimate in bpm, or ``None`` if not yet
                available.
        """
        self._rgb.append((r, g, b))
        if len(self._rgb) >= max(2, int(self.min_fill * self._win)):
            self._recompute()
        return self.hr

    def update(self, frame_bgr, box=None, roi_mask=None,
               frame_time: Optional[float] = None) -> Optional[float]:
        """Ingest one frame; recompute HR once the window is filled.

        Provide a ``roi_mask`` (boolean, frame-sized -- e.g. a SegFace skin mask)
        for segmentation-based sampling, or a face ``box`` to use the built-in
        forehead/cheek skin ROI. ``roi_mask`` takes precedence. If neither yields a
        usable sample the window is left unchanged.

        Args:
            frame_bgr (np.ndarray): BGR frame of shape ``(H, W, 3)``.
            box (Sequence[float], optional): Face box ``(x1, y1, x2, y2)`` in
                pixels. Used only when ``roi_mask`` is ``None``. Defaults to
                ``None``.
            roi_mask (np.ndarray, optional): Boolean frame-sized ROI mask; takes
                precedence over ``box``. Defaults to ``None``.
            frame_time (float, optional): Timestamp of the frame in seconds.
                Accepted for API symmetry; not required for estimation. Defaults to
                ``None``.

        Returns:
            float | None: The current HR estimate in bpm, or ``None`` if not yet
                available.
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
        """Snapshot the current estimates.

        Returns:
            dict: ``{"hr": bpm, "snr": dB, "method": name}`` where ``hr`` and
                ``snr`` may be ``None`` until the window fills.
        """
        return {"hr": self.hr, "snr": self.snr, "method": self.method_name}

    # ------------------------------------------------------------ RR / HRV / RR
    def rri(self, correct: bool = False):
        """RR-interval (inter-beat) series from the current BVP window.

        Detects systolic peaks in the latest band-passed :attr:`bvp` and returns the
        inter-beat intervals -- the substrate for HRV. No extra buffering: this reads
        the same sliding window used for HR.

        Args:
            correct (bool, optional): Clean the series with the Lipponen-Tarvainen
                artefact corrector before returning. Defaults to ``False``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(rri_ms, t_sec)`` inter-beat intervals in
                milliseconds and their timestamps in seconds (empty until enough beats
                are present). When ``correct`` is ``True`` the timestamps are the
                cumulative RR times of the cleaned series.

        See Also:
            [`bvp_to_rri`][physiotrack.signals.bvp_to_rri],
            [`correct_rr_artifacts`][physiotrack.signals.correct_rr_artifacts].
        """
        rri_ms, t_sec = bvp_to_rri(self.bvp, self.fps, self.hr_band)
        if correct and rri_ms.size >= 4:
            rri_ms, _ = correct_rr_artifacts(rri_ms)
            t_sec = np.cumsum(rri_ms) / 1000.0
        return rri_ms, t_sec

    def hrv(self, domains=("time", "frequency", "nonlinear"), correct: bool = True):
        """Heart-rate-variability indices over the current BVP window.

        Computes RR intervals from the sliding window (optionally artefact-corrected)
        and returns the requested HRV indices via
        [`compute_hrv`][physiotrack.signals.compute_hrv].

        Args:
            domains (Sequence[str], optional): Any of ``"time"``, ``"frequency"``,
                ``"nonlinear"``. Defaults to all three.
            correct (bool, optional): Apply artefact correction to the RR series first.
                Defaults to ``True``.

        Returns:
            dict: The HRV indices, or ``{}`` if fewer than ~3 beats are available.

        Note:
            HRV needs a longer window than HR: construct the estimator with a larger
            ``window_sec`` (e.g. ``HeartRateEstimator("POS", fps, window_sec=60)``) for
            stable time- and especially frequency-domain indices. The default 10 s
            window is tuned for HR and yields only coarse HRV.
        """
        rri_ms, t_sec = self.rri(correct=correct)
        if rri_ms.size < 3:
            return {}
        return compute_hrv(rri_ms, t_sec, domains=domains)

    def respiration_rate(self, source: str = "pulse"):
        """Respiration rate (breaths/min) from the current BVP window.

        Args:
            source (str, optional): ``"pulse"`` uses the pulse amplitude modulation
                (RIAV) of :attr:`bvp`; ``"rsa"`` uses RR-interval variation (respiratory
                sinus arrhythmia). Defaults to ``"pulse"``.

        Returns:
            float: Respiration rate in breaths per minute, or ``np.nan`` if unavailable.

        Raises:
            ValueError: If ``source`` is not ``"pulse"`` or ``"rsa"``.

        See Also:
            [`respiration_from_pulse`][physiotrack.signals.respiration_from_pulse],
            [`respiration_from_rri`][physiotrack.signals.respiration_from_rri].
        """
        if source == "pulse":
            _, rr = respiration_from_pulse(self.bvp, self.fps)
            return rr
        if source == "rsa":
            rri_ms, t_sec = self.rri()
            _, rr = respiration_from_rri(rri_ms, t_sec)
            return rr
        raise ValueError(f"Unknown source {source!r}; choose 'pulse' or 'rsa'.")

    def clear(self) -> None:
        """Reset the sliding window and clear the HR/SNR/BVP state."""
        self._rgb.clear()
        self._hr_hist.clear()
        self.bvp = np.array([])
        self.hr = self.snr = None
