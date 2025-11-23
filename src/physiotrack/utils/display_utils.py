"""
Display utilities for video output and screen management.
"""

import cv2
import numpy as np

# For screen size detection
try:
    import tkinter as tk
except ImportError:
    tk = None


def get_screen_size(verbose=False):
    """
    Get screen size for resizing display window.

    Args:
        verbose: Whether to print screen size information

    Returns:
        tuple: (screen_width, screen_height) in pixels
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
            print(f"Display window will fit to screen size: {screen_width}x{screen_height}")

        return screen_width, screen_height

    except Exception as e:
        # Fallback to common screen resolution if detection fails
        screen_width = 1920
        screen_height = 1080
        if verbose:
            print(f"Could not detect screen size, using default: {screen_width}x{screen_height}")

        return screen_width, screen_height


def resize_frame_for_display(frame, screen_width, screen_height):
    """
    Resize frame to fit screen while maintaining aspect ratio.

    Args:
        frame: Input frame (numpy array)
        screen_width: Maximum width for display
        screen_height: Maximum height for display

    Returns:
        numpy array: Resized frame
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
