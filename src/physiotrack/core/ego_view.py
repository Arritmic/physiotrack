"""
Ego video view for overlaying ego-centric video on main output frames.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from physiotrack.core.overlay import OverlayCanvas, alpha_composite
from .panel import PanelMixin


class EgoVideoView(PanelMixin):
    """
    Ego video view for displaying synchronized ego-centric video overlay.
    """

    # Placement and compositing come from PanelMixin; these are this panel's
    # own defaults, preserved exactly as they were before the consolidation.
    PANEL_POSITION = 'bottom_right'
    PANEL_MARGIN = 10
    PANEL_BACKDROP = True
    PANEL_BACKDROP_PAD = 3
    PANEL_BACKDROP_ALPHA = 0.3


    def __init__(self,
                 ego_video_path: str,
                 max_width: int = 320,
                 max_height: int = 240,
                 show_title: bool = True):
        """
        Initialize ego video view.

        Args:
            ego_video_path: Path to the ego video file
            max_width: Maximum width for the ego video canvas
            max_height: Maximum height for the ego video canvas
            show_title: Whether to show "Ego View" title on the canvas
        """
        self.ego_video_path = ego_video_path
        self.max_width = max_width
        self.max_height = max_height
        self.show_title = show_title
        self.enabled = True

        # Open ego video
        self.cap = cv2.VideoCapture(ego_video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open ego video: {ego_video_path}")

        # Get ego video properties
        self.ego_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.ego_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ego_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ego_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate scaled size maintaining aspect ratio
        scale = min(max_width / self.ego_width, max_height / self.ego_height)
        self.canvas_width = int(self.ego_width * scale)
        self.canvas_height = int(self.ego_height * scale)
        self.canvas_size = (self.canvas_width, self.canvas_height)

        # Current frame
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_id = -1

    def read_frame(self, frame_id: int) -> bool:
        """
        Read a specific frame from the ego video.

        Args:
            frame_id: Frame index to read

        Returns:
            True if frame was read successfully, False otherwise
        """
        if frame_id < 0 or frame_id >= self.ego_frame_count:
            return False

        # Only seek if we're not at the next frame
        if frame_id != self.current_frame_id + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = self.cap.read()
        if ret:
            # Resize frame to canvas size
            self.current_frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            self.current_frame_id = frame_id
            return True

        return False

    def get_frame_for_timestamp(self, timestamp: float) -> bool:
        """
        Get ego video frame for a given timestamp.

        Args:
            timestamp: Timestamp in seconds

        Returns:
            True if frame was read successfully
        """
        frame_id = int(timestamp * self.ego_fps)
        return self.read_frame(frame_id)

    def panel_visible(self) -> bool:
        """Nothing is drawn until an ego frame has been read."""
        return bool(self.enabled) and self.current_frame is not None

    def render(self) -> np.ndarray:
        """
        Render the ego video canvas.

        Returns:
            Ego video canvas as numpy array (BGR)
        """
        if self.current_frame is None:
            # Return empty canvas if no frame
            canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas.fill(40)
            if self.show_title:
                ov = OverlayCanvas(self.canvas_width, self.canvas_height)
                ov.text((10, 6), "Ego View", size=18, color=(255, 255, 255), bold=True)
                ov.text((10, 36), "No frame", size=18, color=(128, 128, 128))
                alpha_composite(canvas, ov.render(), 0, 0)
            return canvas

        canvas = self.current_frame.copy()

        # Add title if enabled
        if self.show_title:
            ch, cw = canvas.shape[:2]
            ov = OverlayCanvas(cw, ch)
            ov.text((10, 6), "Ego View", size=20, color=(255, 255, 255), bold=True)
            alpha_composite(canvas, ov.render(), 0, 0)

        return canvas

    def get_canvas_height(self) -> int:
        """Get the current canvas height for stacking calculations."""
        return self.canvas_height

    def release(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        """Destructor to release resources."""
        self.release()
