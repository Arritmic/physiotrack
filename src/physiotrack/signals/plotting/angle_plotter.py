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
>>> plotter.update(result.to_dict()["detections"], frame_time=t)   # per frame
>>> frame = plotter.attach_panels(frame, position="top_left")      # joint + ROM grids
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Sequence

from physiotrack.signals.motion.features import (
    JOINT_ANGLE_TRIPLETS,
    ROM_DEFINITIONS,
    DEFAULT_ROM_MOVEMENTS,
    compute_joint_angle_2d,
    compute_rom_angles,
    rom_color,
)

# Side color coding (BGR) for the interior joint-angle cells.
_LEFT_COLOR = (171, 134, 46)    # #2E86AB
_RIGHT_COLOR = (114, 59, 162)   # #A23B72


class JointAnglePlotter:
    """Render interior joint angles and clinical ROM as 2-column grid panels."""

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
        """Measure the configured angles from one frame's pose results."""
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
        out: Dict[str, float] = {}
        if not keypoints:
            return out
        kp = {k["id"]: k for k in keypoints}
        for joint in self.joints:
            a_id, b_id, c_id = JOINT_ANGLE_TRIPLETS[joint]
            a, b, c = kp.get(a_id), kp.get(b_id), kp.get(c_id)
            if not (a and b and c):
                continue
            if min(a["confidence"], b["confidence"], c["confidence"]) < self.conf_threshold:
                continue
            rad = compute_joint_angle_2d((a["x"], a["y"]), (b["x"], b["y"]), (c["x"], c["y"]))
            if rad is not None and not np.isnan(rad):
                out[joint] = float(np.degrees(rad))
        return out

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

        ``which='joint'`` -> interior joint angles; ``which='rom'`` -> clinical ROM.
        Columns are body side, rows are movement type; each cell is a full plot
        (label, value, gauge, sparkline). Returns None if there is nothing to show.
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
        canvas = np.zeros((h, width, 4), np.uint8)
        cv2.rectangle(canvas, (0, 0), (width - 1, h - 1),
                      (24, 22, 28, int(self.bg_alpha * 255)), -1)
        cv2.rectangle(canvas, (0, 0), (width - 1, h - 1), (105, 95, 90, 220), 1)
        cv2.putText(canvas, title, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (col_w, header_h), (col_w, h - 4), (70, 72, 80, 200), 1)

        for ri, t in enumerate(rows):
            y = header_h + ri * row_h
            for ci, side in enumerate(("L", "R")):
                cell = cells.get((t, side))
                if cell is not None:
                    self._draw_cell(canvas, ci * col_w, y, col_w, f"{side} {t}", *cell)
        return canvas

    def _draw_cell(self, canvas, x, y, cw, label, buf, color):
        """One grid cell: label + value + gauge + (optional) sparkline."""
        c4 = (color[0], color[1], color[2], 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        value = self._latest(buf)

        cv2.putText(canvas, label, (x + 6, y + 14), font, 0.4, c4, 1, cv2.LINE_AA)
        vtxt = f"{value:.0f}" if value is not None else "--"
        (tw, _), _ = cv2.getTextSize(vtxt, font, 0.42, 1)
        cv2.putText(canvas, vtxt, (x + cw - tw - 8, y + 14), font, 0.42, c4, 1, cv2.LINE_AA)

        bx1, bx2, by = x + 6, x + cw - 8, y + 21
        cv2.rectangle(canvas, (bx1, by), (bx2, by + 6), (70, 72, 80, 220), -1)
        if value is not None:
            fx = bx1 + int((bx2 - bx1) * float(np.clip(value / 180.0, 0.0, 1.0)))
            cv2.rectangle(canvas, (bx1, by), (fx, by + 6), c4, -1)

        if self.show_sparkline:
            self._draw_sparkline(canvas, buf, c4, x1=bx1, x2=bx2, y_top=y + 32, height=16)

    @staticmethod
    def _draw_sparkline(canvas, buf, color4, *, x1, x2, y_top, height):
        vals = np.array(buf, dtype=float)
        finite = vals[~np.isnan(vals)]
        if finite.size < 2:
            return
        lo, hi = float(np.min(finite)), float(np.max(finite))
        rng = (hi - lo) or 1.0
        xs = np.linspace(x1, x2, len(vals))
        pts = [[int(x), int(y_top + height - (v - lo) / rng * height)]
               for x, v in zip(xs, vals) if not np.isnan(v)]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color4, 1, cv2.LINE_AA)

    # ----------------------------------------------------------------- compose
    def attach_canvas(self, frame: np.ndarray, canvas: Optional[np.ndarray],
                      position: str = "top_left", margin: int = 10,
                      above_element_height: int = 0) -> np.ndarray:
        """Alpha-composite a BGRA grid canvas onto the frame."""
        if canvas is None:
            return frame
        h, w = frame.shape[:2]
        ch, cw = canvas.shape[:2]
        if cw > w - 2 * margin:
            sc = (w - 2 * margin) / cw
            canvas = cv2.resize(canvas, (int(cw * sc), int(ch * sc)))
            ch, cw = canvas.shape[:2]
        extra = margin if above_element_height > 0 else 0
        y1 = (margin + above_element_height + extra) if "top" in position \
            else (h - ch - margin - above_element_height - extra)
        x1 = margin if "left" in position else w - cw - margin
        if y1 < 0 or x1 < 0 or y1 + ch > h or x1 + cw > w:
            return frame
        roi = frame[y1:y1 + ch, x1:x1 + cw].astype(np.float32)
        alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
        blended = alpha * canvas[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi
        out = frame.copy()
        out[y1:y1 + ch, x1:x1 + cw] = blended.astype(np.uint8)
        return out

    def attach_panels(self, frame: np.ndarray, position: str = "top_left",
                      margin: int = 10, width: Optional[int] = None) -> np.ndarray:
        """Convenience: stack the joint-angle grid then the ROM grid on the frame."""
        width = self.canvas_width if width is None else width
        offset = 0
        for which in ("joint", "rom"):
            grid = self.render_grid(which, width)
            if grid is None:
                continue
            frame = self.attach_canvas(frame, grid, position, margin, offset)
            offset += grid.shape[0] + 10
        return frame

    # ------------------------------------------------------------------ access
    def values(self) -> Dict[str, Optional[float]]:
        """Current interior joint angle (degrees) per configured joint."""
        return {j: self._latest(self._buffers[j]) for j in self.joints}

    def rom_values(self) -> Dict[str, Optional[float]]:
        """Current clinical ROM angle (degrees) per configured movement."""
        return {m: self._latest(self._rom_buffers[m]) for m in self.rom_movements}

    def clear(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        for buf in self._rom_buffers.values():
            buf.clear()
