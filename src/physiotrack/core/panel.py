"""One placement-and-compositing implementation for every side panel.

Seven classes -- the four spatial views, the estimator panels, the keypoint-motion plot
and the joint-angle grids -- each drew themselves into a corner of the frame, and each
carried its own copy of the same corner arithmetic. The copies had drifted: some honoured
a stacking offset and some did not (so a radar view could never be stacked beneath
another panel), some clamped out-of-bounds placement and some raised, and the
alpha-blending was written three different ways.

`PanelMixin` holds that logic once. A panel supplies `render()`; the mixin decides where
it goes and how it is blended.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from .overlay import alpha_composite

__all__ = ["PanelMixin", "attach_stack"]

_VALID_POSITIONS = ("top", "top_left", "top_right",
                    "bottom", "bottom_left", "bottom_right")


class PanelMixin:
    """Corner placement and compositing for a renderable side panel.

    A subclass implements :meth:`render` and inherits :meth:`attach_to_frame`. Two class
    attributes tune the appearance:

    Attributes:
        PANEL_POSITION (str): Default corner, used when the caller does not pass one.
        PANEL_MARGIN (int): Default margin from the frame edge, in pixels.
        PANEL_BACKDROP (bool): Draw a translucent dark box behind the panel so light
            content stays legible over a bright frame. Ignored for BGRA canvases, which
            carry their own alpha.
        PANEL_BACKDROP_PAD (int): How far the backdrop extends past the canvas, in pixels.
        PANEL_BACKDROP_ALPHA (float): Backdrop opacity in ``[0, 1]``. Panels whose content
            is already high-contrast use a lighter value.
    """

    PANEL_POSITION: str = "bottom_right"
    PANEL_MARGIN: int = 10
    PANEL_BACKDROP: bool = False
    PANEL_BACKDROP_PAD: int = 5
    PANEL_BACKDROP_ALPHA: float = 0.3

    def render(self) -> Optional[np.ndarray]:
        """Draw the panel.

        Returns:
            np.ndarray | None: A BGR ``(h, w, 3)`` or BGRA ``(h, w, 4)`` canvas, or
                ``None`` when there is nothing to show yet (too little history, no
                subject). ``None`` leaves the frame untouched.

        Raises:
            NotImplementedError: Always, unless a subclass overrides it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement render() to be attachable."
        )

    def panel_visible(self) -> bool:
        """Whether this panel should appear on the frame at all.

        Separate from :meth:`render` because several panels deliberately render a
        "no data" placeholder for direct callers, while the pipeline should draw nothing
        at all in that state -- an empty grey box in the corner of every frame reads as a
        broken overlay. Subclasses override this to name the condition (typically an
        ``enabled`` flag plus "has this panel received any data yet").

        Returns:
            bool: ``True`` to composite, ``False`` to leave the frame untouched.
        """
        return True

    def attach_to_frame(self, frame: np.ndarray, position: Optional[str] = None,
                        margin: Optional[int] = None,
                        above_element_height: int = 0) -> np.ndarray:
        """Composite this panel onto a corner of ``frame``.

        Args:
            frame (np.ndarray): Target BGR frame ``(H, W, 3)``. Not modified in place.
            position (str, optional): One of ``"top"``, ``"top_left"``, ``"top_right"``,
                ``"bottom"``, ``"bottom_left"``, ``"bottom_right"``. Bare ``"top"`` and
                ``"bottom"`` mean the left corner. Defaults to ``None`` (the class's
                ``PANEL_POSITION``).
            margin (int, optional): Margin from the frame edge in pixels. Defaults to
                ``None`` (the class's ``PANEL_MARGIN``).
            above_element_height (int, optional): Height in pixels of a panel already
                occupying this corner, so several panels can stack without overlapping.
                Defaults to ``0``.

        Returns:
            np.ndarray: A copy of ``frame`` with the panel composited, or ``frame``
                itself when :meth:`panel_visible` is ``False``, :meth:`render` returned
                ``None``, or the panel cannot fit.

        Raises:
            ValueError: If ``position`` is not one of the accepted values.
        """
        if not self.panel_visible():
            return frame
        canvas = self.render()
        if canvas is None:
            return frame
        return self._place(frame, canvas, position, margin, above_element_height)

    # ------------------------------------------------------------------ internals
    def _place(self, frame: np.ndarray, canvas: np.ndarray,
               position: Optional[str] = None, margin: Optional[int] = None,
               above_element_height: int = 0) -> np.ndarray:
        """Composite an already-rendered ``canvas`` onto ``frame``.

        Split out from :meth:`attach_to_frame` so :func:`attach_stack` can render each
        panel exactly once and still reuse the placement rules.
        """
        position = self.PANEL_POSITION if position is None else position
        margin = self.PANEL_MARGIN if margin is None else margin
        if position not in _VALID_POSITIONS:
            raise ValueError(
                f"Invalid position {position!r}; choose from {list(_VALID_POSITIONS)}."
            )

        canvas = self._fit_width(canvas, frame.shape[1], margin)
        origin = self._origin(frame.shape[:2], canvas.shape[:2], position, margin,
                              above_element_height)
        if origin is None:
            return frame
        return self._composite(frame, canvas, *origin)

    @staticmethod
    def _fit_width(canvas: np.ndarray, frame_width: int, margin: int) -> np.ndarray:
        """Downscale a canvas that is wider than the frame allows, preserving aspect."""
        ch, cw = canvas.shape[:2]
        available = frame_width - 2 * margin
        if cw <= available or available <= 0:
            return canvas
        scale = available / cw
        return cv2.resize(canvas, (int(cw * scale), max(1, int(ch * scale))))

    @staticmethod
    def _origin(frame_shape: Tuple[int, int], canvas_shape: Tuple[int, int],
                position: str, margin: int, above_element_height: int
                ) -> Optional[Tuple[int, int]]:
        """Top-left pixel for the canvas, or ``None`` if it cannot fit.

        A non-zero ``above_element_height`` adds one further ``margin`` of separation, so
        a stacked panel is never flush against the one before it. The total gap between
        two stacked panels is therefore ``gutter + margin`` -- which is what shipped
        before this logic was shared, and is preserved deliberately so existing overlays
        look unchanged.
        """
        h, w = frame_shape
        ch, cw = canvas_shape
        if ch > h - 2 * margin or cw > w - 2 * margin:
            return None

        gutter = margin if above_element_height > 0 else 0
        offset = above_element_height + gutter
        y = margin + offset if "top" in position else h - ch - margin - offset
        x = w - cw - margin if "right" in position else margin
        if y < 0 or x < 0 or y + ch > h or x + cw > w:
            return None
        return int(x), int(y)

    def _composite(self, frame: np.ndarray, canvas: np.ndarray,
                   x: int, y: int) -> np.ndarray:
        """Blend ``canvas`` into a copy of ``frame`` at ``(x, y)``."""
        out = frame.copy()
        ch, cw = canvas.shape[:2]

        if canvas.shape[2] == 4:
            # A BGRA canvas carries its own alpha; a backdrop would fight it.
            alpha_composite(out, canvas, x, y)
            return out

        if self.PANEL_BACKDROP:
            pad = self.PANEL_BACKDROP_PAD
            shade = out.copy()
            cv2.rectangle(shade, (x - pad, y - pad), (x + cw + pad, y + ch + pad),
                          (0, 0, 0), -1)
            alpha = self.PANEL_BACKDROP_ALPHA
            out = cv2.addWeighted(out, 1.0 - alpha, shade, alpha, 0)

        out[y:y + ch, x:x + cw] = canvas
        return out


def attach_stack(frame: np.ndarray, panels, position: Optional[str] = None,
                 margin: Optional[int] = None, gutter: int = 10) -> np.ndarray:
    """Composite several panels into one corner, stacked in order.

    Each panel is rendered once and placed using its own compositing rules, offset by the
    combined height of the panels already placed. Callers used to track that offset by
    hand and ask each panel for its height through a different accessor
    (``canvas_height``, ``canvas_size[1]``, ``get_canvas_height()``); the height is taken
    from the rendered canvas instead, so no panel needs to publish one and the offsets
    cannot fall out of step with what was actually drawn.

    Args:
        frame (np.ndarray): Target BGR frame ``(H, W, 3)``. Not modified in place.
        panels (Iterable[PanelMixin | None]): Panels in stacking order, growing away from
            the chosen corner. ``None`` entries are skipped, so an optional panel can be
            listed unconditionally.
        position (str, optional): Corner for the whole stack. Defaults to ``None``, which
            lets each panel use its own ``PANEL_POSITION``.
        margin (int, optional): Margin from the frame edge. Defaults to ``None`` (each
            panel's ``PANEL_MARGIN``).
        gutter (int, optional): Extra vertical gap between stacked panels, in pixels, on
            top of the one ``margin`` of separation the placement rules already insert.
            The total gap is therefore ``gutter + margin``. Defaults to ``10``, matching
            the spacing the overlays used before this logic was shared.

    Returns:
        np.ndarray: The frame with every renderable panel composited.

    Example:
        ```python
        from physiotrack.core.panel import attach_stack

        # radar at the corner, depth above it, ego video above that
        frame = attach_stack(frame, [radar_view, depth_view, ego_view], "bottom_right")
        ```
    """
    offset = 0
    for panel in panels:
        if panel is None or not panel.panel_visible():
            continue
        canvas = panel.render()
        if canvas is None:
            continue
        frame = panel._place(frame, canvas, position, margin, offset)
        offset += canvas.shape[0] + gutter
    return frame
