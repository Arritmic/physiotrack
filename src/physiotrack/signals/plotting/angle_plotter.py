"""Joint-angle & ROM grid overlay for video processing.

``JointAnglePlotter`` measures the major anatomical joint angles (interior angles
at the shoulders, elbows, hips and knees) and, optionally, clinical range-of-motion
(ROM) movements (hip flexion / extension / abduction / adduction), and renders them
as compact, semi-transparent **2-column (left | right) grids** that composite onto a
video frame. Each cell is a small plot: label, live value (deg), a 0--180 deg gauge
bar, and a sparkline of the recent trace. Joint angles and ROM are rendered as
separate grids so the two are visually distinct.

The angle/ROM definitions and the per-movement colors are shared with
``physiotrack.signals.motion.features`` so the panel and the skeleton arcs agree.

Example
-------
>>> from physiotrack.signals import JointAnglePlotter
>>> plotter = JointAnglePlotter(rom=True, fps=30.0)
>>> plotter.update(result.to_dict()["instances"], frame_time=t)   # per frame
>>> frame = plotter.attach_to_frame(frame, position="top_left")    # joint + ROM grids
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Sequence

from physiotrack.core.overlay import OverlayCanvas
from physiotrack.core.panel import PanelMixin
from physiotrack.signals.motion.features import (
    JOINT_ANGLE_TRIPLETS,
    ROM_DEFINITIONS,
    DEFAULT_ROM_MOVEMENTS,
    joint_angles,
    compute_rom_angles,
    rom_color,
)

# Side color coding (BGR) for the interior joint-angle cells.
_LEFT_COLOR = (171, 134, 46)    # #2E86AB
_RIGHT_COLOR = (114, 59, 162)   # #A23B72


class JointAnglePlotter(PanelMixin):
    """Render interior joint angles and clinical ROM as 2-column grid overlays.

    Per frame, measures the interior anatomical joint angles (shoulders, elbows, hips,
    knees) and, optionally, clinical range-of-motion movements (hip flexion / extension
    / abduction / adduction), keeps a smoothing window per quantity, and draws them as
    compact semi-transparent 2-column (left | right) grid panels that composite onto a
    video frame. Each cell shows a label, the live value in degrees, a 0-180 deg gauge
    bar and a sparkline of the recent trace. Angle/ROM definitions and per-movement
    colors are shared with
    [`physiotrack.signals.motion.features`][physiotrack.signals.joint_angles] so the
    panels and any skeleton arcs agree.

    Attributes:
        joints (list[str]): Configured interior joints (subset of ``JOINT_ANGLE_TRIPLETS``).
        rom_movements (list[str]): Configured ROM movements (subset of ``ROM_DEFINITIONS``).
        fps (float): Frame rate (used for the time window; falls back to ``30.0``).
        window_size (int): Samples retained per angle for smoothing/sparkline.
        canvas_width (int): Default grid width in pixels.
        conf_threshold (float): Minimum keypoint confidence for a measurement.

    Example:
        ```python
        import cv2
        import physiotrack as pt
        from physiotrack.signals import JointAnglePlotter

        pose = pt.Pose.Person()
        plotter = JointAnglePlotter(rom=True, fps=30.0)   # 8 joints + clinical ROM

        cap = cv2.VideoCapture("in.mp4")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose_results = pose.predict(frame).to_dict()["instances"]
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            plotter.update(pose_results, frame_time=t)
            frame = plotter.attach_to_frame(frame, position="top_left")
            cv2.imshow("angles", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ```

    See Also:
        [`joint_angles`][physiotrack.signals.joint_angles],
        [`compute_rom_angles`][physiotrack.signals.compute_rom_angles]: the underlying
            plotter-free measurement functions.
    """

    _JOINT_TYPE_ORDER = ["Shoulder", "Elbow", "Hip", "Knee"]
    _ROM_TYPE_ORDER = ["Flexion", "Extension", "Abduction", "Adduction"]

    def __init__(self,
                 joints: Optional[Sequence[str]] = None,
                 *,
                 rom=None,
                 fps: float = 30.0,
                 window_size: int = 150,
                 canvas_width: int = 360,
                 conf_threshold: float = 0.3,
                 smooth: bool = True,
                 show_sparkline: bool = True,
                 bg_alpha: float = 0.7):
        """Configure which joints/ROM movements to measure and how to render them.

        Args:
            joints (Sequence[str], optional): Interior joints to measure, keys of
                ``JOINT_ANGLE_TRIPLETS`` (``leftShoulder``, ``rightShoulder``,
                ``leftElbow``, ``rightElbow``, ``leftHip``, ``rightHip``, ``leftKnee``,
                ``rightKnee``). Defaults to ``None`` (all eight).
            rom (bool | Sequence[str], optional): Clinical ROM to add. ``True`` uses the
                default hip set (``DEFAULT_ROM_MOVEMENTS``); ``None``/``False`` disables
                ROM; a sequence selects specific movements from ``ROM_DEFINITIONS``.
                Defaults to ``None``.
            fps (float, optional): Frame rate. Defaults to ``30.0`` (also used as the
                fallback when a non-positive value is passed).
            window_size (int, optional): Samples kept per angle for smoothing and the
                sparkline. Defaults to ``150``.
            canvas_width (int, optional): Default panel width in pixels (min 240 when
                rendered). Defaults to ``360``.
            conf_threshold (float, optional): Minimum keypoint confidence in ``[0, 1]``.
                Defaults to ``0.3``.
            smooth (bool, optional): Report the median of the last 3 samples instead of
                the latest raw value. Defaults to ``True``.
            show_sparkline (bool, optional): Draw the recent-trace sparkline in each cell
                (taller rows). Defaults to ``True``.
            bg_alpha (float, optional): Panel background opacity in ``[0, 1]``. Defaults
                to ``0.7``.

        Raises:
            ValueError: If ``joints`` contains an unknown joint name, or ``rom`` contains
                an unknown ROM movement name.
        """
        if joints is None:
            joints = list(JOINT_ANGLE_TRIPLETS.keys())
        unknown = [j for j in joints if j not in JOINT_ANGLE_TRIPLETS]
        if unknown:
            raise ValueError(f"Unknown joint(s) {unknown}. Valid: {list(JOINT_ANGLE_TRIPLETS)}")
        self.joints = list(joints)

        if rom is True:
            rom_movements = list(DEFAULT_ROM_MOVEMENTS)
        elif rom in (None, False):
            rom_movements = []
        else:
            rom_movements = list(rom)
        bad = [m for m in rom_movements if m not in ROM_DEFINITIONS]
        if bad:
            raise ValueError(f"Unknown ROM movement(s) {bad}. Valid: {list(ROM_DEFINITIONS)}")
        self.rom_movements = rom_movements

        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.window_size = int(window_size)
        self.canvas_width = int(canvas_width)
        self.conf_threshold = float(conf_threshold)
        self.smooth = bool(smooth)
        self.show_sparkline = bool(show_sparkline)
        self.bg_alpha = float(np.clip(bg_alpha, 0.0, 1.0))

        self._buffers: Dict[str, deque] = {j: deque(maxlen=self.window_size) for j in self.joints}
        self._rom_buffers: Dict[str, deque] = {m: deque(maxlen=self.window_size) for m in self.rom_movements}

    # ------------------------------------------------------------------ update
    def update(self, pose_results: List[dict], frame_time: float = 0.0) -> None:
        """Measure the configured angles from one frame's pose results.

        Uses the first person with keypoints, measures each configured joint angle (and
        ROM movement, if enabled) and appends them to the smoothing buffers; ``NaN`` is
        stored for anything not confidently measured this frame.

        Args:
            pose_results (list[dict]): Per-person pose results, each with a
                ``"keypoints"`` list (e.g. ``result.to_dict()["instances"]``).
            frame_time (float, optional): Frame timestamp in seconds. Defaults to ``0.0``.
        """
        keypoints = self._first_keypoints(pose_results)
        angles = self._measure(keypoints)
        for joint in self.joints:
            self._buffers[joint].append(angles.get(joint, np.nan))
        if self.rom_movements:
            rom = compute_rom_angles(keypoints, self.rom_movements,
                                     self.conf_threshold) if keypoints else {}
            for m in self.rom_movements:
                self._rom_buffers[m].append(rom.get(m, np.nan))

    @staticmethod
    def _first_keypoints(pose_results: List[dict]) -> Optional[List[dict]]:
        for pr in pose_results or []:
            kps = pr.get("keypoints") if isinstance(pr, dict) else None
            if kps:
                return kps
        return None

    def _measure(self, keypoints: Optional[List[dict]]) -> Dict[str, float]:
        # The measurement itself lives in signals.motion.features.joint_angles so
        # it can be used directly on keypoints without this plotter; here we just
        # consume it.
        return joint_angles(keypoints, self.joints, self.conf_threshold)

    def _latest(self, buf: deque) -> Optional[float]:
        vals = [v for v in buf if not np.isnan(v)]
        if not vals:
            return None
        if self.smooth and len(vals) >= 3:
            return float(np.median(vals[-3:]))
        return float(vals[-1])

    # ------------------------------------------------------------------- grids
    @staticmethod
    def _parse_joint(name):
        side = "L" if name.startswith("left") else "R"
        return side, (name[4:] if name.startswith("left") else name[5:])

    @classmethod
    def _parse_rom(cls, name):
        side = "L" if name.startswith("left") else "R"
        for kw in cls._ROM_TYPE_ORDER:
            if kw.lower() in name.lower():
                return side, kw
        return side, name

    def render_grid(self, which: str, width: int) -> Optional[np.ndarray]:
        """Render a 2-column (L | R) grid of angle cells to a BGRA canvas.

        Columns are body side (left | right), rows are movement type; each cell is a
        small plot (label, live value in degrees, 0-180 gauge bar and optional
        sparkline).

        Args:
            which (str): ``"joint"`` for interior joint angles, ``"rom"`` for clinical
                ROM movements.
            width (int): Target grid width in pixels (clamped to a minimum of 240).

        Returns:
            numpy.ndarray | None: The grid as a BGRA (4-channel) canvas, or ``None`` if
                there is nothing to show (no configured rows for ``which``).
        """
        if which == "joint":
            title, order = "JOINT ANGLES (deg)", self._JOINT_TYPE_ORDER
            cells = {}
            for j in self.joints:
                side, t = self._parse_joint(j)
                cells[(t, side)] = (self._buffers[j],
                                    _LEFT_COLOR if side == "L" else _RIGHT_COLOR)
        else:
            title, order = "ROM (deg)", self._ROM_TYPE_ORDER
            cells = {}
            for m in self.rom_movements:
                side, t = self._parse_rom(m)
                cells[(t, side)] = (self._rom_buffers[m], rom_color(m))

        rows = [t for t in order if (t, "L") in cells or (t, "R") in cells]
        if not rows:
            return None

        width = max(int(width), 240)
        col_w = width // 2
        header_h = 22
        row_h = 54 if self.show_sparkline else 30
        h = header_h + row_h * len(rows) + 6
        ov = OverlayCanvas(width, h, bg=(24, 22, 28), bg_alpha=self.bg_alpha,
                           border=(105, 95, 90), radius=6)
        ov.text((8, 4), title, size=16, color=(235, 235, 235), bold=True)
        ov.line((col_w, header_h), (col_w, h - 4), (70, 72, 80), width=1)

        for ri, t in enumerate(rows):
            y = header_h + ri * row_h
            for ci, side in enumerate(("L", "R")):
                cell = cells.get((t, side))
                if cell is not None:
                    self._draw_cell(ov, ci * col_w, y, col_w, f"{side} {t}", *cell)
        return ov.render()

    def _draw_cell(self, ov, x, y, cw, label, buf, color):
        """One grid cell: label + value + gauge + (optional) sparkline."""
        value = self._latest(buf)

        ov.text((x + 6, y + 3), label, size=14, color=color)
        vtxt = f"{value:.0f}" if value is not None else "--"
        vw, _ = ov.measure(vtxt, 15, bold=True)
        ov.text((x + cw - vw - 8, y + 2), vtxt, size=15, color=color, bold=True)

        bx1, bx2, by = x + 6, x + cw - 8, y + 21
        ov.rect((bx1, by), (bx2, by + 6), (70, 72, 80), fill=True)
        if value is not None:
            fx = bx1 + int((bx2 - bx1) * float(np.clip(value / 180.0, 0.0, 1.0)))
            ov.rect((bx1, by), (fx, by + 6), color, fill=True)

        if self.show_sparkline:
            self._draw_sparkline(ov, buf, color, x1=bx1, x2=bx2, y_top=y + 32, height=16)

    @staticmethod
    def _draw_sparkline(ov, buf, color, *, x1, x2, y_top, height):
        vals = np.array(buf, dtype=float)
        finite = vals[~np.isnan(vals)]
        if finite.size < 2:
            return
        lo, hi = float(np.min(finite)), float(np.max(finite))
        rng = (hi - lo) or 1.0
        xs = np.linspace(x1, x2, len(vals))
        pts = [(float(x), float(y_top + height - (v - lo) / rng * height))
               for x, v in zip(xs, vals) if not np.isnan(v)]
        if len(pts) >= 2:
            ov.polyline(pts, color, width=1)

    # ----------------------------------------------------------------- compose
    PANEL_POSITION = "top_left"
    PANEL_MARGIN = 10
    PANEL_BACKDROP = False          # the grids are BGRA and carry their own alpha

    def render(self, width: Optional[int] = None) -> Optional[np.ndarray]:
        """Render the joint-angle grid and the ROM grid as one stacked BGRA canvas.

        Both grids belong to the same panel, so they are fused here rather than
        composited separately: the caller places one canvas and the vertical gap between
        the grids is this panel's business, not the caller's.

        Args:
            width (int, optional): Grid width in pixels. Defaults to ``None``
                (``self.canvas_width``).

        Returns:
            np.ndarray | None: A BGRA canvas ``(h, w, 4)``, or ``None`` when neither grid
                has anything to draw yet.
        """
        width = self.canvas_width if width is None else width
        grids = [g for g in (self.render_grid("joint", width),
                             self.render_grid("rom", width)) if g is not None]
        if not grids:
            return None
        if len(grids) == 1:
            return grids[0]

        gap = 10
        w = max(g.shape[1] for g in grids)
        h = sum(g.shape[0] for g in grids) + gap * (len(grids) - 1)
        canvas = np.zeros((h, w, 4), dtype=grids[0].dtype)
        y = 0
        for grid in grids:
            canvas[y:y + grid.shape[0], :grid.shape[1]] = grid
            y += grid.shape[0] + gap
        return canvas

    # ------------------------------------------------------------------ access
    def values(self) -> Dict[str, Optional[float]]:
        """Return the current interior joint angle (degrees) per configured joint.

        Returns:
            dict[str, float | None]: ``{joint_name: degrees}``; the value is ``None`` for
                joints with no confident measurement yet. Reflects the smoothing set by
                ``smooth``.
        """
        return {j: self._latest(self._buffers[j]) for j in self.joints}

    def rom_values(self) -> Dict[str, Optional[float]]:
        """Return the current clinical ROM angle (degrees) per configured movement.

        Returns:
            dict[str, float | None]: ``{movement_name: degrees}``; ``None`` for movements
                with no confident measurement yet.
        """
        return {m: self._latest(self._rom_buffers[m]) for m in self.rom_movements}

    def clear(self) -> None:
        """Empty all angle and ROM smoothing buffers (e.g. between clips)."""
        for buf in self._buffers.values():
            buf.clear()
        for buf in self._rom_buffers.values():
            buf.clear()
