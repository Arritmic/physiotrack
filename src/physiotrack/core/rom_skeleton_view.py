"""ROM skeleton view — a clean white-background skeleton panel with ROM arcs.

Redraws one person's keypoints as a skeleton on a white canvas that matches the
**full frame** (same aspect ratio), so the person's position and movement within
the room are preserved; the canvas is then scaled down to a side panel
(depth-plot sized). The clinical ROM movements are marked as **color-coded arcs**
at the joints (the angle *values* are reported in the joint-angle panel; the arc
color matches that panel row). Same attach/stacking interface as ``DepthView``.
"""

import math
import cv2
import numpy as np
from typing import List, Optional, Tuple

from physiotrack.core.overlay import OverlayCanvas, alpha_composite
from physiotrack.signals.motion.features import (
    ROM_DEFINITIONS, compute_joint_angle_2d, rom_color,
)

# COCO-17 body skeleton edges (head omitted for clarity).
_COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class ROMSkeletonView:
    """Render one person's skeleton + color-coded ROM arcs on a side panel."""

    def __init__(self,
                 max_width: int = 320,
                 max_height: int = 600,
                 conf_threshold: float = 0.3,
                 show_title: bool = True):
        self.max_width = max_width
        self.max_height = max_height
        self.conf_threshold = float(conf_threshold)
        self.show_title = show_title
        self.enabled = True
        self.canvas: Optional[np.ndarray] = None
        self.canvas_size: Tuple[int, int] = (max_width, max_height)

    # ------------------------------------------------------------------ update
    def update(self, keypoints: Optional[List[dict]], movements: List[str],
               frame_shape) -> None:
        """Build the skeleton canvas (full-room scaled) with ROM arcs."""
        self.canvas = None
        if not keypoints or frame_shape is None:
            return
        H, W = int(frame_shape[0]), int(frame_shape[1])
        if H <= 0 or W <= 0:
            return
        kp = {k["id"]: k for k in keypoints if k.get("confidence", 0.0) >= self.conf_threshold}
        if len(kp) < 3:
            return

        scale = min(self.max_width / W, self.max_height / H)
        cw, ch = max(1, int(W * scale)), max(1, int(H * scale))
        canvas = np.full((ch, cw, 3), 255, np.uint8)
        # Skeleton + arcs are drawn on a supersampled overlay for crisp anti-aliased
        # bones/arcs, then composited onto the white panel.
        ov = OverlayCanvas(cw, ch)

        def to_px(x, y):
            return (int(x * scale), int(y * scale))

        # Skeleton — thin gray bones + small joints (kept light so the colored
        # ROM arcs stand out).
        for a, b in _COCO_EDGES:
            if a in kp and b in kp:
                ov.line(to_px(kp[a]["x"], kp[a]["y"]),
                        to_px(kp[b]["x"], kp[b]["y"]), (205, 205, 205), width=1)
        for k in kp.values():
            ov.circle(to_px(k["x"], k["y"]), 2, (150, 150, 150), fill=True)

        # Color-coded ROM arcs (no value text — values live in the angle panel).
        for name in movements:
            spec = ROM_DEFINITIONS.get(name)
            if spec is None:
                continue
            v, r, m = kp.get(spec["vertex"]), kp.get(spec["ref"]), kp.get(spec["moving"])
            if not (v and r and m):
                continue
            rad = compute_joint_angle_2d((r["x"], r["y"]), (v["x"], v["y"]), (m["x"], m["y"]))
            if rad is None or np.isnan(rad):
                continue
            color = rom_color(name)
            c, mp, rp = to_px(v["x"], v["y"]), to_px(m["x"], m["y"]), to_px(r["x"], r["y"])
            ov.line(c, mp, color, width=2)   # moving segment
            ov.line(c, rp, color, width=2)   # reference axis
            radius = max(6, int(math.hypot(mp[0] - c[0], mp[1] - c[1]) * 0.30))
            self._draw_arc(ov, c, self._vec_angle(c, rp), self._vec_angle(c, mp), radius, color)

        alpha_composite(canvas, ov.render(), 0, 0)
        self.canvas = canvas
        self.canvas_size = (cw, ch)

    @staticmethod
    def _vec_angle(center, point) -> float:
        return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0])) % 360

    @staticmethod
    def _draw_arc(ov, center, ang_ref, ang_vec, radius, color):
        a1, a2 = ang_ref % 360, ang_vec % 360
        d = (a2 - a1 + 360) % 360
        start, end = (a1, a1 + d) if d <= 180 else (a2, a2 + (360 - d))
        ov.arc(center, radius, start, end, color, width=2)

    # ------------------------------------------------------------------ render
    def render(self) -> np.ndarray:
        if self.canvas is None:
            eh = self.max_height // 3
            canvas = np.full((eh, self.max_width, 3), 245, np.uint8)
            ov = OverlayCanvas(self.max_width, eh)
            ov.text((10, 6), "ROM skeleton", size=20, color=(60, 60, 60), bold=True)
            ov.text((10, 34), "No person", size=18, color=(130, 130, 130))
            alpha_composite(canvas, ov.render(), 0, 0)
            return canvas
        canvas = self.canvas.copy()
        if self.show_title:
            ch, cw = canvas.shape[:2]
            ov = OverlayCanvas(cw, ch)
            ov.text((8, 4), "ROM skeleton", size=18, color=(40, 40, 40), bold=True)
            alpha_composite(canvas, ov.render(), 0, 0)
        return canvas

    def attach_to_frame(self, frame: np.ndarray, position: str = 'top_left',
                        margin: int = 10, above_element_height: int = 0) -> np.ndarray:
        if not self.enabled or self.canvas is None:
            return frame
        canvas = self.render()
        h, w = frame.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        extra = (margin if above_element_height > 0 else 0)
        if position == 'bottom_right':
            y1 = h - canvas_h - margin - above_element_height - extra
            x1 = w - canvas_w - margin
        elif position == 'bottom_left':
            y1 = h - canvas_h - margin - above_element_height - extra
            x1 = margin
        elif position == 'top_right':
            y1 = margin + above_element_height + extra
            x1 = w - canvas_w - margin
        elif position == 'top_left':
            y1 = margin + above_element_height + extra
            x1 = margin
        else:
            raise ValueError(f"Invalid position: {position}")
        y2, x2 = y1 + canvas_h, x1 + canvas_w
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            return frame

        result = frame.copy()
        overlay = result.copy()
        cv2.rectangle(overlay, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 0), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        result[y1:y2, x1:x2] = canvas
        return result

    def get_canvas_height(self) -> int:
        return self.canvas.shape[0] if self.canvas is not None else self.max_height
