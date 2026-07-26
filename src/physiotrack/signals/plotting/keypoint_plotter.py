"""
Real-time keypoint motion plotter for tracking keypoint movements during video processing.
Plots keypoint positions relative to pelvis (normalized) with optional filtering.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional, Tuple, List, Dict
from scipy.signal import butter, lfilter

from physiotrack.core.overlay import OverlayCanvas, alpha_composite, SS
import warnings
from ...core.panel import PanelMixin


class KeypointMotionPlotter(PanelMixin):
    """Real-time overlay of one keypoint's motion, measured relative to the pelvis.

    Tracks a single COCO keypoint per frame, expresses its position relative to the
    pelvis (hip midpoint) so the trace is translation-invariant, keeps a sliding window
    of the recent X/Y trajectory, optionally band-pass filters it, and renders it as a
    small Matplotlib panel that can be composited onto a video frame. Currently follows
    the first person with valid keypoints.

    Attributes:
        keypoint_id (int): COCO keypoint id being tracked.
        keypoint_name (str): Display name shown as the panel title.
        window_size (int): Number of frames retained in the sliding window.
        fps (float): Frame rate, used for the time axis and filter design.
        filter_signal (bool): Whether the band-pass filter is active.
        x_buffer (collections.deque): Recent pelvis-relative X positions.
        y_buffer (collections.deque): Recent pelvis-relative Y positions.
        time_buffer (collections.deque): Timestamps aligned with the buffers.

    Example:
        ```python
        import cv2
        import physiotrack as pt
        from physiotrack.signals import KeypointMotionPlotter

        pose = pt.Pose.Person()
        plotter = KeypointMotionPlotter(keypoint_id=9, keypoint_name="left_wrist", fps=30.0)

        cap = cv2.VideoCapture("in.mp4")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose_results = pose.predict(frame).to_dict()["instances"]
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            plotter.update(pose_results, frame_time=t)
            frame = plotter.attach_to_frame(frame, position="top_right")
            cv2.imshow("motion", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ```

    Note:
        The ``Video`` orchestrator wires this up automatically via
        ``plot_keypoint=<id>`` / ``plot_keypoint_name=<name>``.
    """

    # Placement and compositing come from PanelMixin; these are this panel's
    # own defaults, preserved exactly as they were before the consolidation.
    PANEL_POSITION = 'top_right'
    PANEL_MARGIN = 10
    PANEL_BACKDROP = True
    PANEL_BACKDROP_PAD = 5
    PANEL_BACKDROP_ALPHA = 0.15


    def __init__(self,
                 keypoint_id: int = 9,
                 keypoint_name: str = "left_wrist",
                 window_size: int = 300,
                 canvas_width: int = 450,
                 canvas_height: int = 180,
                 filter_signal: bool = True,
                 filter_bandpass: Tuple[float, float] = (0.5, 5.0),
                 fps: float = 30.0):
        """Initialize the keypoint motion plotter and (optionally) its band-pass filter.

        Args:
            keypoint_id (int, optional): COCO keypoint id to track (``9`` = left wrist,
                ``10`` = right wrist, ...). Defaults to ``9``.
            keypoint_name (str, optional): Display name shown as the panel title.
                Defaults to ``"left_wrist"``.
            window_size (int, optional): Number of frames held in the sliding window.
                Defaults to ``300``.
            canvas_width (int, optional): Plot canvas width in pixels. Defaults to ``450``.
            canvas_height (int, optional): Plot canvas height in pixels. Defaults to ``180``.
            filter_signal (bool, optional): Apply a Butterworth band-pass filter to the
                trace. Defaults to ``True``. Silently disabled if ``fps <= 0`` or the
                filter fails to initialize.
            filter_bandpass (tuple[float, float], optional): Band-pass ``(low, high)``
                cutoff frequencies in Hz. Defaults to ``(0.5, 5.0)``.
            fps (float, optional): Video frame rate, used for the time axis and filter
                design. Defaults to ``30.0``.
        """
        self.keypoint_id = keypoint_id
        self.keypoint_name = keypoint_name
        self.window_size = window_size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.filter_signal = filter_signal
        self.filter_bandpass = filter_bandpass
        self.fps = fps
        
        # Data buffers (relative to pelvis)
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)
        
        # Track colors for each tracked person
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}
        
        # Cache for matplotlib figure/axes (reuse for performance)
        self._fig = None
        self._ax = None
        self._line_x = None
        self._line_y = None
        
        # Initialize filter coefficients if filtering is enabled
        if self.filter_signal and fps > 0:
            self._init_filter()
        
    def _init_filter(self):
        """Initialize Butterworth bandpass filter."""
        try:
            order = 3
            self.filter_b, self.filter_a = butter(
                order, 
                self.filter_bandpass, 
                btype='bandpass', 
                fs=self.fps
            )
        except Exception as e:
            warnings.warn(f"Could not initialise the signal filter: {e!r}. "
                          f"Plotting the unfiltered signal.", RuntimeWarning, stacklevel=2)
            self.filter_signal = False
    
    def _get_pelvis_position(self, keypoints: List[dict]) -> Optional[Tuple[float, float]]:
        """
        Calculate pelvis position as the midpoint between left and right hips.
        
        Args:
            keypoints: List of keypoints [{'id': int, 'x': float, 'y': float, 'confidence': float}, ...]
        
        Returns:
            tuple | None: Pelvis position ``(x, y)``, or ``None`` if the hips are not visible.
        """
        if not keypoints:
            return None
        
        # Create lookup dict
        kp_dict = {kp['id']: kp for kp in keypoints}
        
        # COCO keypoint IDs: 11=left_hip, 12=right_hip
        left_hip = kp_dict.get(11)
        right_hip = kp_dict.get(12)
        
        confidence_threshold = 0.3
        
        if left_hip and right_hip:
            if left_hip['confidence'] > confidence_threshold and right_hip['confidence'] > confidence_threshold:
                pelvis_x = (left_hip['x'] + right_hip['x']) / 2.0
                pelvis_y = (left_hip['y'] + right_hip['y']) / 2.0
                return (pelvis_x, pelvis_y)
        
        return None
    
    def _get_keypoint_position(self, keypoints: List[dict], keypoint_id: int) -> Optional[Tuple[float, float, float]]:
        """
        Get position and confidence of a specific keypoint.
        
        Returns:
            tuple | None: ``(x, y, confidence)`` for the keypoint, or ``None`` if not found.
        """
        if not keypoints:
            return None
        
        for kp in keypoints:
            if kp['id'] == keypoint_id:
                if kp['confidence'] > 0.3:
                    return (kp['x'], kp['y'], kp['confidence'])
        
        return None
    
    def update(self, pose_results: List[dict], frame_time: float):
        """Ingest one frame's pose results and append the tracked point to the buffers.

        Finds the pelvis (hip midpoint) and the tracked keypoint for the first valid
        person, computes the keypoint position relative to the pelvis, and appends it to
        the sliding buffers. Frames without a confident pelvis or tracked keypoint are
        skipped.

        Args:
            pose_results (list[dict]): Per-person pose results, each with a
                ``"keypoints"`` list of ``{"id", "x", "y", "confidence"}`` dicts (e.g.
                ``result.to_dict()["instances"]``).
            frame_time (float): Current frame timestamp in seconds.
        """
        # For now, track the first person with valid keypoints
        # TODO: Extend to multi-person tracking
        
        for pose_result in pose_results:
            if 'keypoints' not in pose_result or pose_result['keypoints'] is None:
                continue
            
            keypoints = pose_result['keypoints']
            
            # Get pelvis position (reference point)
            pelvis = self._get_pelvis_position(keypoints)
            if pelvis is None:
                continue
            
            # Get tracked keypoint position
            kp_data = self._get_keypoint_position(keypoints, self.keypoint_id)
            if kp_data is None:
                continue
            
            kp_x, kp_y, kp_conf = kp_data
            
            # Calculate position relative to pelvis (normalized)
            rel_x = kp_x - pelvis[0]
            rel_y = kp_y - pelvis[1]
            
            # Add to buffers
            self.x_buffer.append(rel_x)
            self.y_buffer.append(rel_y)
            self.time_buffer.append(frame_time)
            
            # Only track first valid person for now
            break
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current motion window as a BGR plot image.

        Plots the pelvis-relative X and Y traces (band-pass filtered when enabled and
        long enough), with an auto-scaled y-axis and a rolling time window on x. The
        Matplotlib figure is created once and reused for speed, rendered supersampled
        and area-downscaled to the exact canvas size for crisp output.

        Returns:
            numpy.ndarray: The plot as an ``(canvas_height, canvas_width, 3)`` BGR array.
                Before ~10 samples are collected, a "Collecting motion data..." canvas is
                returned instead.
        """
        if len(self.x_buffer) < 10:
            # Not enough data to plot
            return self._render_empty_canvas()
        
        # Convert buffers to arrays
        x_data = np.array(self.x_buffer)
        y_data = np.array(self.y_buffer)
        time_data = np.array(self.time_buffer)
        
        # Apply filtering if enabled
        if self.filter_signal and len(x_data) > 20:
            try:
                x_filtered = lfilter(self.filter_b, self.filter_a, x_data)
                y_filtered = lfilter(self.filter_b, self.filter_a, y_data)
            except Exception as e:
                # If filtering fails, use unfiltered data
                x_filtered = x_data
                y_filtered = y_data
        else:
            x_filtered = x_data
            y_filtered = y_data
        
        # Create or reuse cached figure for performance
        if self._fig is None:
            # Render at SS x the target DPI (same figure size in inches, so the
            # point-sized fonts/lines stay identical) then area-downscale to the
            # panel size -- supersampled anti-aliasing, so the plot is crisp instead
            # of the blurry low-DPI render.
            self._fig, self._ax = plt.subplots(
                figsize=(self.canvas_width / 80, self.canvas_height / 80), dpi=80 * SS)
            
            # Initial plot setup
            self._line_x, = self._ax.plot([], [], label='X', linewidth=1.2, color='#2E86AB', antialiased=True)
            self._line_y, = self._ax.plot([], [], label='Y', linewidth=1.2, color='#A23B72', antialiased=True)
            
            # Styling (only set once)
            self._ax.set_xlabel('Time (s)', fontsize=8)
            self._ax.set_ylabel('Rel. to pelvis (px)', fontsize=8)
            self._ax.set_title(f'{self.keypoint_name}', fontsize=9, fontweight='bold', pad=5)
            self._ax.legend(loc='upper right', fontsize=7, frameon=False)
            self._ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            self._ax.tick_params(labelsize=7)
            self._fig.tight_layout()
        
        # Update data (much faster than recreating plot)
        self._line_x.set_data(time_data, x_filtered)
        self._line_y.set_data(time_data, y_filtered)
        
        # Update axis limits
        if len(time_data) > 0:
            time_window = self.window_size / self.fps
            self._ax.set_xlim(time_data[-1] - time_window, time_data[-1])
            
            # Auto-scale y-axis based on visible data
            y_min = min(np.min(x_filtered), np.min(y_filtered))
            y_max = max(np.max(x_filtered), np.max(y_filtered))
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 10
            self._ax.set_ylim(y_min - margin, y_max + margin)
        
        # Render to numpy array
        self._fig.canvas.draw()
        
        # Get buffer data (compatible with newer matplotlib versions)
        try:
            # Try newer API first
            buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to BGR for OpenCV
            canvas = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            # Fallback for older matplotlib versions
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            # Convert RGB to BGR for OpenCV
            canvas = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        # Resize to exact canvas dimensions (area-average the SS-supersampled render)
        if canvas.shape[:2] != (self.canvas_height, self.canvas_width):
            canvas = cv2.resize(canvas, (self.canvas_width, self.canvas_height),
                                interpolation=cv2.INTER_AREA)
        
        return canvas
    
    def _render_empty_canvas(self) -> np.ndarray:
        """Render an empty canvas with a waiting message."""
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 245

        ov = OverlayCanvas(self.canvas_width, self.canvas_height)
        text = "Collecting motion data..."
        tw, th = ov.measure(text, 22, bold=True)
        ov.text(((self.canvas_width - tw) / 2, (self.canvas_height - th) / 2), text,
                size=22, color=(100, 100, 100), bold=True)
        alpha_composite(canvas, ov.render(), 0, 0)
        return canvas
    
    def clear(self):
        """Clear all buffered data and close the cached Matplotlib figure.

        Empties the X/Y/time buffers and releases the reused figure/axes so a fresh plot
        is built on the next render. Call between independent clips.
        """
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.time_buffer.clear()
        
        # Clean up matplotlib figure
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._line_x = None
            self._line_y = None

