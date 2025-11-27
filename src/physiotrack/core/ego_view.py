"""
Ego video view for overlaying ego-centric video on main output frames.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class EgoVideoView:
    """
    Ego video view for displaying synchronized ego-centric video overlay.
    """

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
                cv2.putText(canvas, "Ego View", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(canvas, "No frame", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            return canvas

        canvas = self.current_frame.copy()

        # Add title if enabled
        if self.show_title:
            cv2.putText(canvas, "Ego View", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return canvas

    def attach_to_frame(self, frame: np.ndarray, position: str = 'bottom_right',
                        margin: int = 10, above_element_height: int = 0) -> np.ndarray:
        """
        Attach ego video view to a video frame.

        Args:
            frame: Video frame to attach ego view to
            position: Position on frame ('bottom_right', 'bottom_left', 'top_right', 'top_left')
            margin: Margin from frame edge in pixels
            above_element_height: Height of element below this one to stack above it

        Returns:
            Frame with ego view attached
        """
        if not self.enabled or self.current_frame is None:
            return frame

        canvas = self.render()
        h, w = frame.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]

        # Calculate position based on specified location
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
        cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 0), -1)
        result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)

        # Overlay ego view
        result_frame[y1:y2, x1:x2] = canvas

        return result_frame

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
