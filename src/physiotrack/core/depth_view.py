"""
Depth view for visualizing depth estimation results overlaid on video frames.
Provides a miniature depth map display similar to the radar view.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class DepthView:
    """
    Depth view visualization for displaying depth estimation results.

    Renders a colorized depth map as a miniature overlay on video frames,
    positioned similar to the radar/floor map view.
    """

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
            cv2.putText(canvas, "Depth Map", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, "No data", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            return canvas

        canvas = self.depth_canvas.copy()

        # Add title if enabled
        if self.show_title:
            cv2.putText(canvas, "Depth Map", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return canvas

    def attach_to_frame(self, frame: np.ndarray, position: str = 'bottom_right',
                        margin: int = 10, above_element_height: int = 0) -> np.ndarray:
        """
        Attach depth view to a video frame.

        Args:
            frame: Video frame to attach depth view to
            position: Position on frame ('bottom_right', 'bottom_left', 'top_right', 'top_left')
            margin: Margin from frame edge in pixels
            above_element_height: Height of element below this one (e.g., radar view) to stack above it

        Returns:
            Frame with depth view attached
        """
        if not self.enabled or self.depth_canvas is None:
            return frame

        canvas = self.render()
        h, w = frame.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]

        # Calculate position based on specified location
        # Stack above any existing element (like radar view)
        if position == 'bottom_right':
            y1 = h - canvas_h - margin - above_element_height - (margin if above_element_height > 0 else 0)
            y2 = y1 + canvas_h
            x1 = w - canvas_w - margin
            x2 = w - margin
        elif position == 'bottom_left':
            y1 = h - canvas_h - margin - above_element_height - (margin if above_element_height > 0 else 0)
            y2 = y1 + canvas_h
            x1 = margin
            x2 = margin + canvas_w
        elif position == 'top_right':
            y1 = margin + above_element_height + (margin if above_element_height > 0 else 0)
            y2 = y1 + canvas_h
            x1 = w - canvas_w - margin
            x2 = w - margin
        elif position == 'top_left':
            y1 = margin + above_element_height + (margin if above_element_height > 0 else 0)
            y2 = y1 + canvas_h
            x1 = margin
            x2 = margin + canvas_w
        else:
            raise ValueError(f"Invalid position: {position}")

        # Ensure coordinates are valid
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            return frame

        result_frame = frame.copy()

        # Add semi-transparent background
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 0), -1)
        result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)

        # Overlay depth view
        result_frame[y1:y2, x1:x2] = canvas

        return result_frame

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
            print(f"Warning: Unknown colormap '{colormap}'. Using 'inferno'.")
            self.colormap = 'inferno'
