"""
Display utilities for video output and screen management.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .._logging import get_logger

logger = get_logger(__name__)

# For screen size detection
try:
    import tkinter as tk
except ImportError:
    tk = None


def get_screen_size(verbose: bool = False) -> Tuple[int, int]:
    """Get the usable screen size for sizing a display window.

    Args:
        verbose (bool): Log the detected size at DEBUG level. Defaults to ``False``.

    Returns:
        tuple[int, int]: ``(screen_width, screen_height)`` in pixels, at 90% of the
            detected screen to leave room for the taskbar. Falls back to
            ``(1920, 1080)`` when Tk is unavailable or detection fails.
    """
    try:
        if tk is not None:
            root = tk.Tk()
            root.withdraw()  # Hide the window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            # Use 90% of screen size to leave room for taskbar/menus
            screen_width = int(screen_width * 0.9)
            screen_height = int(screen_height * 0.9)
        else:
            # Fallback to common screen resolution
            screen_width = 1920
            screen_height = 1080

        if verbose:
            logger.debug("Display window fits screen: %dx%d", screen_width, screen_height)

        return screen_width, screen_height

    except Exception as e:
        # Fallback to common screen resolution if detection fails
        screen_width = 1920
        screen_height = 1080
        if verbose:
            logger.debug("Could not detect screen size; using default %dx%d",
                     screen_width, screen_height)

        return screen_width, screen_height


def resize_frame_for_display(frame: np.ndarray,
                             screen_width: Optional[int],
                             screen_height: Optional[int]) -> np.ndarray:
    """Resize a frame to fit the screen while preserving aspect ratio.

    Args:
        frame (np.ndarray): Input frame, ``(H, W, 3)`` BGR.
        screen_width (int, optional): Maximum display width in pixels. ``None``
            disables resizing.
        screen_height (int, optional): Maximum display height in pixels. ``None``
            disables resizing.

    Returns:
        np.ndarray: The frame, downscaled to fit. Frames already smaller than the
            screen are returned unchanged — this never upscales.
    """
    if screen_width is None or screen_height is None:
        return frame

    frame_height, frame_width = frame.shape[:2]

    # Calculate scaling factor to fit within screen
    width_scale = screen_width / frame_width
    height_scale = screen_height / frame_height
    scale = min(width_scale, height_scale)

    # Only resize if frame is larger than screen
    if scale < 1.0:
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame

    return frame
