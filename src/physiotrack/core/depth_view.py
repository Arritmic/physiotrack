"""
Depth view for visualizing depth estimation results overlaid on video frames.
Provides a miniature depth map display similar to the radar view.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from physiotrack.core.overlay import OverlayCanvas, alpha_composite
import warnings
from .panel import PanelMixin


class DepthView(PanelMixin):
    """
    Depth view visualization for displaying depth estimation results.

    Renders a colorized depth map as a miniature overlay on video frames,
    positioned similar to the radar/floor map view.
    """

    # Placement and compositing come from PanelMixin; these are this panel's
    # own defaults, preserved exactly as they were before the consolidation.
    PANEL_POSITION = 'bottom_right'
    PANEL_MARGIN = 10
    PANEL_BACKDROP = True
    PANEL_BACKDROP_PAD = 5
    PANEL_BACKDROP_ALPHA = 0.3


    def __init__(self,
                 max_width: int = 320,
                 max_height: int = 240,
                 colormap: str = 'inferno',
                 show_title: bool = True):
        """
        Initialize depth view.

        Args:
            max_width: Maximum width for the depth view canvas
            max_height: Maximum height for the depth view canvas
            colormap: Colormap to use for depth visualization
                     ('inferno', 'viridis', 'magma', 'plasma', 'jet', 'turbo')
            show_title: Whether to show "Depth Map" title on the canvas
        """
        self.max_width = max_width
        self.max_height = max_height
        self.colormap = colormap
        self.show_title = show_title
        self.enabled = True

        # Current depth canvas (will be set by update())
        self.depth_canvas: Optional[np.ndarray] = None
        self.canvas_size: Tuple[int, int] = (max_width, max_height)

        # Colormap mapping
        self.colormap_dict = {
            'inferno': cv2.COLORMAP_INFERNO,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'magma': cv2.COLORMAP_MAGMA,
            'plasma': cv2.COLORMAP_PLASMA,
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'bone': cv2.COLORMAP_BONE,
            'turbo': cv2.COLORMAP_TURBO,
        }

    def update(self, depth_map: np.ndarray) -> None:
        """
        Update the depth view with a new depth map.

        Args:
            depth_map: Raw depth map (HxW numpy array, any dtype)
        """
        if depth_map is None:
            self.depth_canvas = None
            return

        # Normalize depth to 0-255
        depth_normalized = self._normalize_depth(depth_map)

        # Apply colormap
        cv_colormap = self.colormap_dict.get(self.colormap.lower(), cv2.COLORMAP_INFERNO)
        colored_depth = cv2.applyColorMap(depth_normalized, cv_colormap)

        # Resize to fit max dimensions while maintaining aspect ratio
        h, w = colored_depth.shape[:2]
        scale = min(self.max_width / w, self.max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        self.depth_canvas = cv2.resize(colored_depth, (new_w, new_h))
        self.canvas_size = (new_w, new_h)

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-255 range."""
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_normalized = np.zeros_like(depth)
        return depth_normalized.astype(np.uint8)

    def panel_visible(self) -> bool:
        """Nothing is drawn until a depth map has arrived."""
        return bool(self.enabled) and self.depth_canvas is not None

    def render(self) -> np.ndarray:
        """
        Render the depth view canvas.

        Returns:
            Depth canvas as numpy array (BGR)
        """
        if self.depth_canvas is None:
            # Return empty canvas if no depth data
            canvas = np.zeros((self.max_height, self.max_width, 3), dtype=np.uint8)
            canvas.fill(40)
            ov = OverlayCanvas(self.max_width, self.max_height)
            ov.text((10, 6), "Depth Map", size=22, color=(255, 255, 255), bold=True)
            ov.text((10, 36), "No data", size=18, color=(128, 128, 128))
            alpha_composite(canvas, ov.render(), 0, 0)
            return canvas

        canvas = self.depth_canvas.copy()

        # Add title if enabled
        if self.show_title:
            ch, cw = canvas.shape[:2]
            ov = OverlayCanvas(cw, ch)
            ov.text((10, 6), "Depth Map", size=20, color=(255, 255, 255), bold=True)
            alpha_composite(canvas, ov.render(), 0, 0)

        return canvas

    def get_canvas_height(self) -> int:
        """Get the current canvas height for stacking calculations."""
        if self.depth_canvas is not None:
            return self.depth_canvas.shape[0]
        return self.max_height

    def set_colormap(self, colormap: str) -> None:
        """
        Change the colormap used for depth visualization.

        Args:
            colormap: Colormap name ('inferno', 'viridis', 'magma', 'plasma', 'jet', 'turbo')
        """
        if colormap.lower() in self.colormap_dict:
            self.colormap = colormap.lower()
        else:
            warnings.warn(f"Unknown colormap {colormap!r}; using 'inferno'.",
                          RuntimeWarning, stacklevel=2)
            self.colormap = 'inferno'
