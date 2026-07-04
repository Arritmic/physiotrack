import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from tqdm import tqdm
from physiotrack.modules.Yolo.classes_and_palettes import COLORS
from physiotrack.core.overlay import draw_label
from physiotrack.core.radar_view import RadarView
from physiotrack.core.depth_view import DepthView
from physiotrack.core.ego_view import EgoVideoView
from physiotrack.core.rom_skeleton_view import ROMSkeletonView
from physiotrack.signals.plotting.keypoint_plotter import KeypointMotionPlotter
from physiotrack.signals.plotting.angle_plotter import JointAnglePlotter
from physiotrack.capture.orientation import resolve_rotation, apply_rotation
from physiotrack.signals.motion.features import DEFAULT_ROM_MOVEMENTS, ROM_DEFINITIONS
from physiotrack.utils import get_screen_size, resize_frame_for_display


class Video:
    """High-level orchestrator that runs the full inference pipeline over a video.

    ``Video`` ties together the individual predictors -- detection, pose
    estimation, tracking, segmentation, face detection / orientation and depth --
    and drives them frame-by-frame (optionally in batches) across a clip, camera
    device or RTSP stream. It composites every enabled model's output onto each
    frame, builds the requested side panels (keypoint-motion plot, joint-angle /
    ROM grids, ROM skeleton canvas, top-down radar view, depth view, ego-video
    view), writes an annotated output video, and returns / saves the per-frame
    results as structured data.

    You attach models by passing already-constructed predictor instances to the
    constructor (``detector``, ``pose``, ``tracker`` ...); any left as ``None`` are
    simply skipped. Nothing runs until you call [`run`][physiotrack.Video.run].
    Every stage is optional, so the same class covers a bare pose-only pass as well
    as the complete multi-model pipeline.

    The pipeline order per batch is: detection -> pose -> tracking -> segmentation
    -> face orientation -> depth -> overlay/compositing. When both a custom
    ``detector`` and a ``pose`` estimator are supplied, the pose estimator must be
    a ``Pose.Custom`` instance (so it consumes external boxes); otherwise a
    ``ValueError`` is raised at run time.

    Attributes:
        video_path (str | Path | int): The original ``source`` argument.
        source_identifier (str): Human-readable id derived from the source (file
            stem, RTSP host, or ``"CAM_device_<n>"``); used to name outputs/logs.
        total_frames (int | None): Frame count for seekable files, else ``None``
            (streams / cameras).
        video_fps (int): Native frames-per-second of the source (falls back to
            ``30`` when it cannot be read).
        width (int): Effective output frame width in pixels (already accounts for
            ``orient`` 90/270 dimension swap).
        height (int): Effective output frame height in pixels.
        batch_size (int): Number of frames processed per batch (>= 1).
        detectors (list): Detector instances (empty if none supplied).
        segmentators (list): Segmentation instances (empty if none supplied).
        pose_estimator: The attached [`Pose`][physiotrack.Pose] predictor or ``None``.
        tracker: The attached [`Tracker`][physiotrack.Tracker] or ``None``.
        output_path (Path): Directory where outputs are written.
        rom_movements (list[str]): Resolved list of enabled ROM movement names.

    Example:
        ```python
        import physiotrack as pt

        # Pose-only pass over a clip (see examples/pose_video.py)
        pose = pt.Pose.VRStudent(verbose=False, device=0)
        video = pt.Video(source="clip.mp4", pose=pose, output_dir="output", verbose=True)
        results = video.run("output/clip_poses.mp4", "output/clip_result.json")
        print(f"Processed {len(results)} frames")
        ```

    See Also:
        [`Detection`][physiotrack.Detection]: person / object detectors to pass as ``detector``.
        [`Pose`][physiotrack.Pose]: pose estimators to pass as ``pose``.
        [`Tracker`][physiotrack.Tracker]: multi-object tracker to pass as ``tracker``.
        [`Result`][physiotrack.Result]: per-frame result object produced by the predictors.
    """

    def __init__(self,
                 source: Union[str, Path, int],
                 *,
                 detector=None,
                 pose=None,
                 segmenter=None,
                 tracker=None,
                 face=None,
                 face_orientation=None,
                 depth=None,
                 ego_video: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 fps: Optional[int] = None,
                 resize: Optional[Tuple[int, int]] = None,
                 rotate: bool = False,
                 orient=0,
                 floor_map: Optional[List[Tuple[int, int]]] = None,
                 floor_map_background: Optional[Union[str, np.ndarray]] = None,
                 floor_map_rotation: int = 90,
                 depth_colormap: str = 'inferno',
                 plot_keypoint: Optional[int] = None,
                 plot_keypoint_name: Optional[str] = None,
                 plot_angles: bool = False,
                 angle_joints: Optional[List[str]] = None,
                 rom=None,
                 rom_render: bool = True,
                 verbose: bool = False,
                 show_fps: bool = False,
                 show: bool = False,
                 batch_size: int = 1):
        """Configure the pipeline: attach models, panels and I/O options.

        Opens the ``source`` immediately (to read fps / resolution / frame count)
        and builds any side-panel views that were requested, but does not process
        any frames -- call [`run`][physiotrack.Video.run] to start. All model
        arguments accept an already-constructed predictor instance (or ``None`` to
        skip that stage). ``detector`` and ``segmenter`` additionally accept a
        ``list`` of instances to run several of the same kind.

        Args:
            source (str | Path | int): Video source. A file path, an ``rtsp://``
                URL, or an integer camera-device index (e.g. ``0``). RTSP streams
                and camera devices have no known frame count.
            detector (optional): Detection predictor, or a ``list`` of them, whose
                boxes feed pose/segmentation/tracking. Defaults to ``None`` (no
                detection). See [`Detection`][physiotrack.Detection]. When a
                detector is combined with a pose estimator, the pose estimator must
                be a ``Pose.Custom``.
            pose (optional): Pose estimator producing keypoints per frame. Defaults
                to ``None``. See [`Pose`][physiotrack.Pose].
            segmenter (optional): Segmentation predictor, or a ``list`` of them,
                whose masks are colorized and alpha-blended onto the frame.
                Defaults to ``None``.
            tracker (optional): Multi-object tracker assigning stable IDs to
                detections; runs frame-by-frame (no batching). Defaults to
                ``None``. See [`Tracker`][physiotrack.Tracker].
            face (optional): Face detector; required for ``face_orientation``.
                Defaults to ``None``.
            face_orientation (optional): Head-pose (yaw/pitch/roll) estimator;
                effective only when ``face`` is also provided. Defaults to ``None``.
            depth (optional): Monocular depth estimator; enables the depth side
                view. Defaults to ``None``.
            ego_video (str | Path, optional): Path to an ego-centric video to
                overlay as a synchronized side view. Defaults to ``None``.
            output_dir (str | Path, optional): Directory for outputs; created if
                missing. Defaults to ``None`` (uses the current working directory).
            fps (int, optional): Target processing frame rate. Frames are subsampled
                so roughly this many frames per source-second are processed and the
                output video is written at this rate. Defaults to ``None`` (process
                every frame at the source fps).
            resize (tuple[int, int], optional): ``(width, height)`` to resize every
                frame to before inference and output. Defaults to ``None`` (keep
                native resolution).
            rotate (bool): If ``True``, rotate every frame 90 degrees clockwise
                (via ``cv2.ROTATE_90_CLOCKWISE``) during preprocessing, swapping the
                output dimensions. Applied after ``orient``. Defaults to ``False``.
            orient (int | str, optional): Explicit orientation fix applied to every
                frame, one of ``0`` / ``90`` / ``180`` / ``270`` degrees (also
                accepts ``None`` / ``"none"`` for no rotation); unknown values fall
                back to ``0``. Use this for phone clips whose display rotation lives
                in container metadata rather than the pixels. ``90``/``270`` swap the
                effective ``width``/``height``. Defaults to ``0`` (no rotation).
            floor_map (list[tuple[int, int]], optional): Four ``(x, y)`` image
                corner points delimiting the floor region; enables the top-down
                radar view (requires both ``tracker`` and ``pose`` at run time).
                Defaults to ``None`` (radar view disabled).
            floor_map_background (str | np.ndarray, optional): Background for the
                radar canvas. ``None`` / ``"default"`` uses a plain canvas;
                ``"auto"`` / ``"extract"`` warps the floor region out of the first
                frame via homography; a path string or image array supplies a
                pre-made floor plan. Defaults to ``None``.
            floor_map_rotation (int): Rotation in degrees (``0`` / ``90`` / ``180``
                / ``270``) applied to orient the radar view. Defaults to ``90``.
            depth_colormap (str): Matplotlib colormap name for the depth view, e.g.
                ``"inferno"``, ``"viridis"``, ``"magma"``, ``"plasma"`` or ``"jet"``.
                Only used when ``depth`` is provided. Defaults to ``"inferno"``.
            plot_keypoint (int, optional): COCO keypoint id to plot as a live motion
                signal in a top-right panel (e.g. ``9`` = left wrist, ``10`` = right
                wrist). Requires a pose estimator at run time. Defaults to ``None``
                (no motion plot).
            plot_keypoint_name (str, optional): Display label for the motion plot.
                Defaults to ``None`` (falls back to ``"keypoint_<id>"``).
            plot_angles (bool): If ``True``, overlay a live joint-angle panel
                (interior joint angles) on the left. Ignored with a warning if no
                pose estimator is provided. Defaults to ``False``.
            angle_joints (list[str], optional): Subset of joint names to show in the
                angle panel, e.g. ``["leftElbow", "rightElbow", "leftKnee",
                "rightKnee"]``. Only used when ``plot_angles`` is ``True``. Defaults
                to ``None`` (shows all available joints).
            rom (bool | list[str], optional): Clinical range-of-motion overlays.
                ``True`` enables the default ROM movement set; a ``list`` selects
                specific movement names (filtered against the known ROM definitions);
                ``None`` / ``False`` disables ROM. Requires a pose estimator.
                Defaults to ``None``.
            rom_render (bool): If ``True`` (and ``rom`` is enabled with a pose
                estimator), draw the white full-room skeleton canvas with
                color-coded ROM arcs. Defaults to ``True``.
            verbose (bool): If ``True``, print setup/progress info and show a tqdm
                progress bar. Defaults to ``False``.
            show_fps (bool): If ``True``, print real-time and end-of-run
                component-wise FPS statistics. Defaults to ``False``.
            show (bool): If ``True``, display the annotated frames in a live OpenCV
                window during processing (press ``q`` to quit). Defaults to ``False``.
            batch_size (int): Number of frames processed together per pipeline step;
                values below ``1`` are clamped to ``1``. Tracking always runs
                frame-by-frame regardless. Defaults to ``1``.

        Raises:
            ValueError: If ``source`` cannot be opened by OpenCV.

        Example:
            ```python
            import physiotrack as pt

            # Pose + explicit detector + tracker (see examples/tracker_aided_pose_video.py)
            pose = pt.Pose.Custom(model=pt.Models.Pose.ViTPose.WholeBody.b_wholebody, device=0)
            detector = pt.Detection.VRStudent(device=0)
            cfg = pt.TrackerConfig(); cfg.tracker_type = "ocsort"; cfg.classes = [0]
            tracker = pt.Tracker(config=cfg)

            video = pt.Video(
                source="clip.mp4",
                pose=pose,
                detector=detector,
                tracker=tracker,
                output_dir="output",
                batch_size=4,
                verbose=True,
            )
            video.run("output/clip_poses.mp4", "output/clip_result.json")
            ```

        Note:
            Constructing the object opens the video capture and (when
            ``plot_keypoint`` is set) briefly opens a second capture to read the fps.
        """

        self.video_path = source
        # Support both single instance and list of instances for detector and segmenter
        self.detectors = detector if isinstance(detector, list) else ([detector] if detector is not None else [])
        self.segmentators = segmenter if isinstance(segmenter, list) else ([segmenter] if segmenter is not None else [])
        self.tracker = tracker
        self.pose_estimator = pose
        self.face_detector = face
        self.face_orientation = face_orientation
        self.depth_estimator = depth
        self.ego_video_path = ego_video
        self.verbose = verbose
        self.show_fps = show_fps
        self.show_output = show
        self.required_fps = fps
        self.frame_resize = resize
        self.frame_rotate = rotate
        self.floor_map = floor_map
        self.batch_size = max(1, batch_size)  # Ensure batch size is at least 1

        # Get screen size for display if show_output is enabled
        self.screen_width = None
        self.screen_height = None
        if self.show_output:
            self.screen_width, self.screen_height = get_screen_size(verbose=self.verbose)

        # Initialize radar view with background mode and rotation
        self.radar_view = RadarView(
            floor_map=floor_map,
            background=floor_map_background,
            rotation=floor_map_rotation
        ) if floor_map else None

        # Initialize depth view if depth estimator is provided
        # Match width to radar view if available, otherwise use default
        self.depth_view = None
        if depth is not None:
            depth_max_width = 320  # default
            if self.radar_view is not None:
                depth_max_width = self.radar_view.canvas_size[0]
            self.depth_view = DepthView(
                max_width=int(depth_max_width * 1.2),  # ~20% larger (matches ROM skeleton)
                max_height=720,  # Allow taller to preserve aspect ratio
                colormap=depth_colormap,
                show_title=True
            )

        # Initialize ego video view if ego video path is provided
        # Match width to radar view if available, otherwise use default
        self.ego_view = None
        if ego_video is not None:
            ego_max_width = 320  # default
            if self.radar_view is not None:
                ego_max_width = self.radar_view.canvas_size[0]
            self.ego_view = EgoVideoView(
                ego_video_path=str(ego_video),
                max_width=ego_max_width,
                max_height=600,  # Allow taller to preserve aspect ratio
                show_title=True
            )

        # Initialize keypoint motion plotter
        self.motion_plotter = None
        if plot_keypoint is not None:
            # Get video FPS for the plotter
            cap_temp = cv2.VideoCapture(source)
            video_fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
            if not video_fps > 0:
                video_fps = 30
            cap_temp.release()
            
            if plot_keypoint_name is None:
                plot_keypoint_name = f"keypoint_{plot_keypoint}"
            
            self.motion_plotter = KeypointMotionPlotter(
                keypoint_id=plot_keypoint,
                keypoint_name=plot_keypoint_name,
                window_size=300,  # 10 seconds at 30fps
                canvas_width=450,  # Reduced width for better performance
                canvas_height=180,  # Reduced height for better performance
                filter_signal=True,
                filter_bandpass=(0.5, 5.0),
                fps=float(video_fps)
            )

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {source}")
        
        self._setup_source_info()
        
        if output_dir:
            self.output_path = Path(output_dir)
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path = Path.cwd()
        
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.video_fps > 0:
            self.video_fps = 30  # Default FPS
            if self.verbose:
                print(f'Using default FPS: {self.video_fps}')
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Orientation fix: opt-in, default 0 (none). Phones store display rotation as
        # container metadata rather than baking it into the pixels, so frames can decode
        # sideways/upside down; pass orient=90/180/270 to rotate when a clip needs it.
        # There is no auto/metadata mode (it is unreliable across builds). Applied to
        # every frame in preprocess_frame; 90/270 swaps the effective dimensions.
        self._rotation = resolve_rotation(orient)
        if self._rotation:
            print(f"[orientation] rotating frames {self._rotation} deg (orient).")
            if self._rotation in (90, 270):
                self.width, self.height = self.height, self.width

        rom_enabled = bool(rom) and rom is not False
        self.rom_movements = []
        if rom_enabled:
            self.rom_movements = (list(DEFAULT_ROM_MOVEMENTS) if rom is True
                                  else [m for m in rom if m in ROM_DEFINITIONS])

        # Joint-angle panel (top-left): interior joint angles (plot_angles) and/or
        # the clinical ROM angle *values* as color-coded rows (rom). The same colors
        # mark the arcs on the skeleton canvas below.
        self.angle_plotter = None
        if plot_angles or rom_enabled:
            if self.pose_estimator is None:
                if self.verbose:
                    print("plot_angles/rom ignored: no pose estimator provided.")
            else:
                self.angle_plotter = JointAnglePlotter(
                    joints=angle_joints if plot_angles else [],
                    rom=self.rom_movements if rom_enabled else None,
                    fps=float(self.video_fps),
                )

        # Skeleton canvas (left, under the angle panel): the person's skeleton on a
        # white full-room canvas with color-coded ROM arcs. Shown when rom is on and
        # rom_render is True.
        self.rom_skeleton_view = None
        if rom_enabled and rom_render and self.rom_movements and self.pose_estimator is not None:
            rom_base_width = self.radar_view.canvas_size[0] if self.radar_view is not None else 320
            self.rom_skeleton_view = ROMSkeletonView(
                max_width=int(rom_base_width * 1.2), max_height=720, show_title=True  # ~20% larger
            )

        if self.verbose:
            print(f"Video properties: {self.width}x{self.height}, {self.video_fps} FPS")
            print(f"Source: {self.source_identifier}")
            print(f"Batch size: {self.batch_size}")
    
    def _setup_source_info(self):
        """Setup source identifier and total frames based on video path type."""
        if isinstance(self.video_path, str):
            path_prefix = ''.join(letter for letter in str(self.video_path).split(':')[0] if letter.isalnum())
            if path_prefix == 'rtsp':
                if self.verbose:
                    print(f'Start processing RTSP stream {self.video_path}')
                source_name = ".".join(self.video_path.split('@')[-1].split('.')[:-1]).replace(':', '-').replace('/', '_')
                self.source_identifier = f'{source_name}'
                self.total_frames = None
            else:
                if self.verbose:
                    print(f'Start processing video {self.video_path}')
                source_name = Path(self.video_path).stem
                self.source_identifier = f'{source_name}'
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.total_frames = frame_count if frame_count > 0 else None
        elif isinstance(self.video_path, int):
            if self.verbose:
                print(f'Start processing camera device {self.video_path}')
            self.source_identifier = f'CAM_device_{self.video_path}'
            self.total_frames = None
        else:
            self.source_identifier = 'unknown_source'
            self.total_frames = None

    @staticmethod
    def select_frames(camera_fps: int, required_fps: Optional[int]) -> List[int]:
        """Choose which 1-based frame positions (within each source-second) to process.

        Given the source frame rate and a desired processing rate, returns the
        indices (in the range ``1..camera_fps``) that should be kept so that
        roughly ``required_fps`` frames are processed per second of video. Used by
        [`run`][physiotrack.Video.run] to subsample frames.

        Args:
            camera_fps (int): Native frames-per-second of the source.
            required_fps (int, optional): Desired processing rate. If ``None`` or
                equal to ``camera_fps``, every frame in the second is selected.

        Returns:
            list[int]: The 1-based frame positions to process within each second.

        Example:
            ```python
            import physiotrack as pt
            pt.Video.select_frames(30, 5)  # -> 5 evenly spaced indices in 1..30
            ```
        """
        if required_fps == camera_fps or required_fps is None:
            return [int(i) for i in np.arange(1, camera_fps+1, dtype=float)]

        delta = camera_fps - 1
        step = delta / required_fps
        y = np.arange(0, required_fps, dtype=float) * step + 1
        return [int(i) for i in y]

    def preprocess_frame(self, frame):
        """Apply orientation, resize and 90-degree rotation to a raw decoded frame.

        Applies, in order: the explicit ``orient`` rotation, the ``resize`` target
        size, and the ``rotate`` 90-degree clockwise rotation -- each only if that
        option was enabled in the constructor. Called on every frame read from the
        source before inference.

        Args:
            frame (np.ndarray): A decoded BGR frame of shape ``(H, W, 3)``.

        Returns:
            np.ndarray: The preprocessed BGR frame. Its dimensions may differ from
                the input when ``orient`` is 90/270, ``resize`` is set, or
                ``rotate`` is ``True``.
        """
        if self._rotation:
            frame = apply_rotation(frame, self._rotation)
        if self.frame_resize:
            frame = cv2.resize(frame, self.frame_resize)
        if self.frame_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
    
    def process_batch_detections(self, frames_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Run all configured detectors over a batch of frames.

        Uses the detector's batched API (``detect_batch``) when available, else
        falls back to per-frame ``detect`` calls, drawing each detector's boxes in
        its own palette color. With no detectors configured, returns the frames
        unchanged with empty detection lists.

        Args:
            frames_batch (list[np.ndarray]): BGR frames of shape ``(H, W, 3)``.

        Returns:
            list[tuple[np.ndarray, list[np.ndarray]]]: One ``(combined_frame,
                all_detections)`` tuple per input frame. ``combined_frame`` is the
                frame with detection boxes drawn; ``all_detections`` is a list with
                one ``(N, 6)`` array per detector, each row
                ``(x1, y1, x2, y2, conf, cls)``.
        """
        batch_results = []
        
        if len(self.detectors) == 0:
            # No detectors, return empty results for each frame
            for frame in frames_batch:
                batch_results.append((frame, []))
            return batch_results
        
        # Prefer batched detection if available
        if len(self.detectors) > 0 and hasattr(self.detectors[0], 'detect_batch'):
            det = self.detectors[0]
            det_outputs = det.detect_batch(frames_batch)
            batch_results = []
            color_names = list(COLORS.keys())
            for frame, (results, detected_frame) in zip(frames_batch, det_outputs):
                detections = results[0].boxes.data.cpu().numpy()  # match single-frame schema
                combined_frame = frame.copy()
                color = tuple(COLORS[color_names[0]])
                for det_box in (detections[:, :4].astype(int) if detections.size else []):
                    x1, y1, x2, y2 = det_box
                    cv2.rectangle(combined_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                batch_results.append((combined_frame, [detections]))
            return batch_results

        # Process each frame in batch (fallback)
        for frame in frames_batch:
            all_detections = []
            combined_frame = frame.copy()
            
            # Use predefined colors from COLORS palette
            color_names = list(COLORS.keys())
            
            for idx, detector in enumerate(self.detectors):
                # YOLO can handle single frame or batch - we pass single for now
                # TODO: Update to pass batch directly when YOLO batch inference is confirmed
                results, detected_frame = detector.detect(frame)
                detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
                
                # Get color for this detector
                color_name = color_names[idx % len(color_names)]
                color = tuple(COLORS[color_name])
                
                # Draw boxes with unique color for this detector
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(combined_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    # Add label with detector index
                    label = f"D{idx}-C{int(cls)}: {conf:.2f}"
                    draw_label(combined_frame, (int(x1), int(y1) - 20), label,
                               size=18, color=color, bold=True)
                
                all_detections.append(detections)
            
            batch_results.append((combined_frame, all_detections))
        
        return batch_results
    
    def process_batch_pose(self, frames_batch: List[np.ndarray], boxes_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, Any]]:
        """Run the pose estimator over a batch of frames and draw keypoints.

        Passes the batch (and per-frame detection boxes, if any) to the pose
        estimator's ``predict``, which returns one [`Result`][physiotrack.Result]
        per frame. If no pose estimator is configured, frames pass through with
        ``None`` results.

        Args:
            frames_batch (list[np.ndarray]): BGR frames of shape ``(H, W, 3)``,
                typically already annotated with detection boxes.
            boxes_batch (list[np.ndarray]): Per-frame person boxes to pose on, each
                an ``(N, 4)`` int array, or ``None`` for a frame with no boxes.

        Returns:
            list[tuple[np.ndarray, Any]]: One ``(result_frame, pose_results)`` tuple
                per frame. ``result_frame`` has keypoints drawn on it;
                ``pose_results`` is the ``detections`` list from the result's
                ``to_dict()`` (per-person keypoints and metadata).
        """
        batch_results = []
        
        if self.pose_estimator is None:
            for frame in frames_batch:
                batch_results.append((frame, None))
            return batch_results
        
        # Pose predict() accepts a list and returns one Result per frame.
        results = self.pose_estimator.predict(frames_batch, boxes_batch)
        for frame, result in zip(frames_batch, results):
            pose_results = result.to_dict()['detections']
            # Draw keypoints onto the (detection-annotated) frame.
            result_frame = result.plot(boxes=False, labels=False, keypoints=True)
            batch_results.append((result_frame, pose_results))

        return batch_results
    
    def process_batch_segmentation(self, frames_batch: List[np.ndarray], 
                                  all_detections_batch: List[List[np.ndarray]],
                                  overlay_frames: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Run all segmentators over a batch and alpha-blend their masks onto frames.

        Gathers each segmentator's segmentation map (using its batched API when
        available), optionally restricts masks to detector boxes when the
        segmentator has ``bbox_filter`` enabled, colorizes and merges the maps
        across segmentators, then blends the result over the base frame (70/30).
        With no segmentators configured, returns ``frames_batch`` unchanged.

        Args:
            frames_batch (list[np.ndarray]): Clean BGR frames used for segmentation
                inference, shape ``(H, W, 3)``.
            all_detections_batch (list[list[np.ndarray]]): Per-frame detection
                arrays (one ``(N, 6)`` array per detector) used for optional
                bbox-based mask filtering.
            overlay_frames (list[np.ndarray], optional): Frames the colorized masks
                are composited onto (e.g. frames already carrying pose/tracking
                drawings). Defaults to ``None`` (composite onto ``frames_batch``).

        Returns:
            list[np.ndarray]: One BGR frame per input with the merged, colorized
                segmentation overlay blended in.
        """
        batch_results = []

        if len(self.segmentators) == 0:
            return frames_batch

        # Prepare colors for merging
        color_names = list(COLORS.keys())
        color_list = [COLORS[name] for name in color_names]

        # First, gather segmentation maps per segmentator, using segment_batch if available
        base_frames_for_overlay = overlay_frames if overlay_frames is not None else frames_batch
        seg_maps_per_segmentator = []  # List[List[np.ndarray]] indexed [seg_idx][frame_idx]
        for seg_idx, segmentator in enumerate(self.segmentators):
            # Try wrapper's batch API, then inner segmentor batch API, else per-frame
            batched_outputs = None

            if hasattr(segmentator, 'segmentor') and hasattr(segmentator.segmentor, 'segment_batch'):
                batched_outputs = segmentator.segmentor.segment_batch(frames_batch)

            seg_maps_for_this_segmentator = []
            if batched_outputs is not None:
                # Expect list of (seg_img, seg_map)
                for (_, seg_map) in batched_outputs:
                    seg_maps_for_this_segmentator.append(seg_map)
            else:
                # Fallback: per-frame segmentation without filtering; we'll filter below
                for frame in frames_batch:
                    seg_map = segmentator.predict(frame).seg_map
                    seg_maps_for_this_segmentator.append(seg_map)

            # Apply optional bbox filtering per frame, matching single-frame logic
            for frame_idx, seg_map in enumerate(seg_maps_for_this_segmentator):
                filter_bboxes = None
                all_detections = all_detections_batch[frame_idx]
                if getattr(segmentator, 'bbox_filter', False) and len(all_detections) > 0:
                    if segmentator.detector_index is not None and segmentator.detector_index < len(all_detections):
                        detections = all_detections[segmentator.detector_index]
                        if segmentator.detector_class_filter is not None:
                            class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                            class_mask = np.isin(detections[:, 5], class_filter)
                            filter_bboxes = detections[class_mask][:, :4] if np.any(class_mask) else None
                        else:
                            filter_bboxes = detections[:, :4]
                    else:
                        all_dets = np.vstack(all_detections) if len(all_detections) > 0 else None
                        if all_dets is not None:
                            if segmentator.detector_class_filter is not None:
                                class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                                class_mask = np.isin(all_dets[:, 5], class_filter)
                                filter_bboxes = all_dets[class_mask][:, :4] if np.any(class_mask) else None
                            else:
                                filter_bboxes = all_dets[:, :4]

                if filter_bboxes is not None and len(filter_bboxes) > 0:
                    h, w = seg_map.shape[:2]
                    filtered_map = np.zeros((h, w), dtype=seg_map.dtype)
                    for bbox in filter_bboxes:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        filtered_map[y1:y2, x1:x2] = seg_map[y1:y2, x1:x2]
                    seg_maps_for_this_segmentator[frame_idx] = filtered_map

            seg_maps_per_segmentator.append(seg_maps_for_this_segmentator)

        # Now merge per-frame across segmentators and overlay
        for frame_idx, frame in enumerate(frames_batch):
            h, w = frame.shape[:2]
            combined_segmentation_map = np.zeros((h, w), dtype=np.uint8)
            combined_segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

            for seg_idx, seg_maps in enumerate(seg_maps_per_segmentator):
                seg_map = seg_maps[frame_idx]
                seg_mask = seg_map > 0
                if not np.any(seg_mask):
                    continue

                # Remap class IDs and colorize
                remapped_map = seg_map.copy()
                remapped_map[seg_mask] = (seg_idx * 100) + seg_map[seg_mask]

                unique_classes = np.unique(seg_map[seg_mask])
                seg_colored = np.zeros_like(combined_segmentation_img)
                for cls_id in unique_classes:
                    if cls_id > 0:
                        cls_mask = seg_map == cls_id
                        color_idx = (seg_idx * 10 + int(cls_id)) % len(color_list)
                        seg_colored[cls_mask] = color_list[color_idx]

                combined_segmentation_map[seg_mask] = remapped_map[seg_mask]
                combined_segmentation_img[seg_mask] = seg_colored[seg_mask]

            base_frame = base_frames_for_overlay[frame_idx]
            result_frame = cv2.addWeighted(base_frame, 0.7, combined_segmentation_img, 0.3, 0)
            batch_results.append(result_frame)

        return batch_results
    
    def process_batch_face_orientation(self, frames_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, List[Dict]]]:
        """Detect faces and estimate head orientation over a batch, drawing pose axes.

        Runs the face detector per frame, then batch head-pose estimation on frames
        that contain faces, and draws yaw/pitch/roll axes at each face center.
        Returns the frames unchanged (with empty results) unless both a ``face``
        detector and a ``face_orientation`` estimator are configured.

        Args:
            frames_batch (list[np.ndarray]): BGR frames of shape ``(H, W, 3)``.

        Returns:
            list[tuple[np.ndarray, list[dict]]]: One ``(result_frame,
                face_orientation_results)`` tuple per frame. Each result dict holds
                a ``"bbox"`` (``x1, y1, x2, y2``) and a ``"pose"`` mapping with
                ``"yaw"``, ``"pitch"`` and ``"roll"`` angles.
        """
        if self.face_detector is None or self.face_orientation is None:
            return [(frame, []) for frame in frames_batch]
        
        batch_results = []
        
        # Detect faces in batch
        face_bboxes_batch = []
        for frame in frames_batch:
            bboxes = self.face_detector.predict(frame).boxes  # (N, 4)
            face_bboxes_batch.append(bboxes if bboxes.size else np.array([]).reshape(0, 4))

        # Estimate face orientation in batch using batch inference
        face_orientation_results_batch = []
        if any(len(bboxes) > 0 for bboxes in face_bboxes_batch):
            # Use batch inference for frames with faces (returns one Result per frame)
            orientation_batch_results = self.face_orientation.predict_batch(frames_batch, face_bboxes_batch)
            for result in orientation_batch_results:
                face_orientation_results_batch.append(result.to_dict()['detections'])
        else:
            face_orientation_results_batch = [[] for _ in frames_batch]

        # Visualize face orientation on frames
        # Reset batch_results to avoid mixing with predict_batch outputs
        batch_results = []
        from physiotrack.face import draw_axis
        for frame, face_bboxes, orientation_results in zip(frames_batch, face_bboxes_batch, face_orientation_results_batch):
            vis_frame = frame.copy()
            if len(orientation_results) > 0:
                for detection in orientation_results:
                    pose = detection['pose']
                    bbox = detection['bbox']

                    x1, y1, x2, y2 = bbox
                    face_center_x = int((x1 + x2) / 2)
                    face_center_y = int((y1 + y2) / 2)

                    face_width = x2 - x1
                    face_height = y2 - y1
                    axis_size = max(face_width, face_height) * 0.6

                    # Draw orientation axes
                    vis_frame = draw_axis(
                        vis_frame,
                        yaw=pose['yaw'],
                        pitch=pose['pitch'],
                        roll=pose['roll'],
                        tdx=face_center_x,
                        tdy=face_center_y,
                        size=axis_size
                    )

            batch_results.append((vis_frame, orientation_results))

        return batch_results

    def process_batch_depth(self, frames_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate a depth map for each frame in a batch.

        Delegates to the depth estimator's ``predict`` (which accepts a list and
        returns one ``DepthResult`` per frame). Returns all ``None`` when no depth
        estimator is configured.

        Args:
            frames_batch (list[np.ndarray]): BGR frames of shape ``(H, W, 3)``.

        Returns:
            list[np.ndarray | None]: One ``(H, W)`` depth map per frame, or ``None``
                entries when depth estimation is disabled.
        """
        if self.depth_estimator is None:
            return [None for _ in frames_batch]

        # predict() accepts a list and returns one DepthResult per frame.
        results = self.depth_estimator.predict(frames_batch)
        return [r.depth for r in results]

    def run(self,
            output_video: Optional[Union[str, Path]] = None,
            output_json: Optional[Union[str, Path]] = None,
            progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Process the whole video and return per-frame results.

        Reads the source frame-by-frame (subsampling to ``fps`` if set), runs every
        configured stage in batches -- detection, pose, tracking, segmentation, face
        orientation, depth -- composites the enabled overlays and side panels, and
        optionally writes an annotated H.264 (``avc1``) video and a JSON dump.
        Blocks until the source is exhausted (or the user presses ``q`` when
        ``show=True``).

        Args:
            output_video (str | Path, optional): Path to write the annotated MP4.
                Defaults to ``None`` (no video written).
            output_json (str | Path, optional): Path to write the per-frame results
                as JSON. Defaults to ``None`` (no JSON written).
            progress_callback (callable, optional): Called after each processed
                frame as ``progress_callback(frame_id, total_frames, pose_results)``.
                Defaults to ``None``.

        Returns:
            list[dict]: One dict per processed frame. Each always has ``frame_id``
                (int) and ``timestamp`` (float seconds), plus, when the relevant
                stage is enabled: ``detections`` (per-person pose results),
                ``track_box`` (the tracked subject box), and ``face_orientation``
                (list of head-pose dicts).

        Raises:
            ValueError: If a custom ``detector`` is configured alongside a pose
                estimator that is not a ``Pose.Custom`` instance.

        Example:
            ```python
            import physiotrack as pt
            pose = pt.Pose.VRStudent(device=0)
            video = pt.Video(source="clip.mp4", pose=pose, output_dir="output")
            results = video.run("output/clip.mp4", "output/clip.json")
            ```

        Note:
            The output video is encoded with the H.264 (``avc1``) codec; its frame
            rate is ``fps`` when set, otherwise the source fps.
        """
        
        pbar = None
        if self.total_frames and self.verbose:
            pbar = tqdm(total=self.total_frames, desc=f'Processing {self.source_identifier}')
        
        selected_frame_ids = self.select_frames(self.video_fps, self.required_fps)
        out_writer = None
        if output_video:
            if self.frame_resize:
                output_width, output_height = self.frame_resize
            else:
                output_width, output_height = self.width, self.height
            if self.frame_rotate:
                output_width, output_height = output_height, output_width

            effective_fps = self.required_fps if self.required_fps else self.video_fps

            # Use avc1 codec for MP4 output
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out_writer = cv2.VideoWriter(str(output_video), fourcc, effective_fps,
                                       (output_width, output_height))

            if self.verbose:
                print(f"Using H.264 (avc1) codec for video encoding")
        
        all_detection_data = []
        frame_count = 0
        frame_filter_count = 1
        start_time = time.time()
        last_fps_print_time = start_time
        fps_print_interval = 2.0  # Print FPS every 2 seconds for real-time monitoring
        
        # Batch collection variables
        frame_batch = []
        frame_batch_metadata = []
        
        # Flag to track if we need to extract floor background from first frame
        extract_floor_from_first_frame = (
            self.radar_view is not None and 
            self.radar_view.background_mode in ["auto", "extract"]
        )

        while True:
            # Collect frames into batch
            batch_ready = False
            
            while len(frame_batch) < self.batch_size:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video, process remaining batch if any
                    if len(frame_batch) > 0:
                        batch_ready = True
                    break
                
                if pbar:
                    pbar.update(1)
                
                video_timestamp = round(frame_count / self.video_fps, 3)
                frame = self.preprocess_frame(frame)
                
                # Extract floor area from first frame if needed
                if extract_floor_from_first_frame and frame_count == 0:
                    if self.verbose:
                        print("Extracting floor area from first frame and transforming to top-down view...")
                    self.radar_view.set_background_from_frame(frame)
                    extract_floor_from_first_frame = False  # Only do this once
                
                # Check if this frame should be processed
                if frame_filter_count in selected_frame_ids:
                    frame_batch.append(frame)
                    frame_batch_metadata.append({
                        'frame_id': frame_count,
                        'timestamp': video_timestamp,
                        'frame_filter_count': frame_filter_count
                    })
                
                frame_count += 1
                frame_filter_count = frame_filter_count + 1 if frame_filter_count < self.video_fps else 1
                
                # Check if batch is full
                if len(frame_batch) == self.batch_size:
                    batch_ready = True
                    break
            
            # Process batch if ready
            if batch_ready and len(frame_batch) > 0:
                # Step 1: Batch detection
                detection_results = self.process_batch_detections(frame_batch)
                
                # Extract frames and detections from results
                frames_with_detections = [r[0] for r in detection_results]
                all_detections_batch = [r[1] for r in detection_results]
                
                # Step 2: Extract boxes for pose estimation
                boxes_batch = []
                for all_detections in all_detections_batch:
                    if len(all_detections) > 0:
                        detections = np.vstack(all_detections)
                        boxes = detections[:, :-2].astype(int)
                        boxes_batch.append(boxes)
                    else:
                        boxes_batch.append(None)
                
                # Step 3: Batch pose estimation (if applicable)
                pose_results_batch = []
                if self.pose_estimator:
                    if self.pose_estimator.__class__.__name__ != "Custom" and len(self.detectors) > 0:
                        raise ValueError("Please use Pose.Custom class if you want to use a custom detector with the Video class.")
                    
                    pose_batch_results = self.process_batch_pose(frames_with_detections, boxes_batch)
                    frames_with_pose = [r[0] for r in pose_batch_results]
                    pose_results_batch = [r[1] for r in pose_batch_results]
                else:
                    frames_with_pose = frames_with_detections
                    pose_results_batch = [[] for _ in frame_batch]
                
                # Step 4: Process tracking frame-by-frame (tracker doesn't support batch)
                frames_with_tracking = []
                online_targets_batch = []
                
                for idx, (frame, all_detections, metadata) in enumerate(zip(frames_with_pose, all_detections_batch, frame_batch_metadata)):
                    if self.tracker is not None and len(all_detections) > 0:
                        detections = np.vstack(all_detections)
                        track_result = self.tracker.track(frame, detections)
                        frame = track_result.rendered
                        online_targets = track_result.raw

                        # Filter boxes based on student track
                        if self.tracker.student_track_id is not None and self.tracker.last_known_bbox is not None:
                            # Update boxes for this frame
                            boxes_batch[idx] = np.array(self.tracker.last_known_bbox, dtype=int).reshape(1, -1)
                    else:
                        online_targets = []
                    
                    frames_with_tracking.append(frame)
                    online_targets_batch.append(online_targets)
                
                # Step 5: Batch segmentation
                # Inference uses clean frames; overlay uses frames_with_tracking to keep pose drawings
                if len(self.segmentators) > 0:
                    frames_after_seg = self.process_batch_segmentation(frame_batch, all_detections_batch, overlay_frames=frames_with_tracking)
                else:
                    frames_after_seg = frames_with_tracking
                
                # Step 6: Batch face orientation processing
                face_orientation_results_batch = []
                if self.face_detector is not None and self.face_orientation is not None:
                    face_orientation_batch_results = self.process_batch_face_orientation(frames_after_seg)
                    frames_with_face_orientation = [r[0] for r in face_orientation_batch_results]
                    face_orientation_results_batch = [r[1] for r in face_orientation_batch_results]
                else:
                    frames_with_face_orientation = frames_after_seg
                    face_orientation_results_batch = [[] for _ in frame_batch]

                # Step 7: Batch depth estimation
                depth_maps_batch = []
                if self.depth_estimator is not None:
                    depth_maps_batch = self.process_batch_depth(frame_batch)
                else:
                    depth_maps_batch = [None for _ in frame_batch]

                # Step 8: Process overlays (radar view, depth view) and save results
                for idx, (result_frame, metadata, pose_results, online_targets, face_orientation_results, depth_map) in enumerate(
                        zip(frames_with_face_orientation, frame_batch_metadata, pose_results_batch, online_targets_batch, face_orientation_results_batch, depth_maps_batch)):
                    
                    # Update and attach motion plotter (top right corner)
                    if self.motion_plotter and self.pose_estimator is not None:
                        self.motion_plotter.update(pose_results, metadata['timestamp'])
                        result_frame = self.motion_plotter.attach_to_frame(result_frame, position='top_right')

                    # (Left-side kinematics stack -- joint-angle grid, ROM grid,
                    #  skeleton -- is composited together after the right-side views.)

                    # Update and attach radar view (bottom right)
                    radar_view_height = 0
                    if self.radar_view and self.tracker is not None and self.pose_estimator is not None:
                        self.radar_view.update(online_targets, pose_results)
                        result_frame = self.radar_view.attach_to_frame(result_frame)
                        radar_view_height = self.radar_view.canvas_size[1] + 10  # height + margin

                    # Update and attach depth view (above radar view on bottom right)
                    depth_view_height = 0
                    if self.depth_view and depth_map is not None:
                        self.depth_view.update(depth_map)
                        result_frame = self.depth_view.attach_to_frame(
                            result_frame,
                            position='bottom_right',
                            margin=10,
                            above_element_height=radar_view_height
                        )
                        depth_view_height = self.depth_view.get_canvas_height() + 10  # height + margin

                    # Update and attach ego video view (above depth view or radar view on bottom right)
                    if self.ego_view is not None:
                        self.ego_view.read_frame(metadata['frame_id'])
                        result_frame = self.ego_view.attach_to_frame(
                            result_frame,
                            position='bottom_right',
                            margin=10,
                            above_element_height=radar_view_height + depth_view_height
                        )

                    # Left-side kinematics stack (top -> bottom): joint-angle grid,
                    # ROM grid (both transparent 2-column L|R panels), then the white
                    # full-room skeleton canvas with color-coded ROM arcs. All share
                    # the skeleton's width so they line up.
                    if self.pose_estimator is not None and (self.angle_plotter is not None
                                                            or self.rom_skeleton_view is not None):
                        first_kps = pose_results[0].get('keypoints') if pose_results else None
                        if self.angle_plotter is not None:
                            self.angle_plotter.update(pose_results, metadata['timestamp'])
                        if self.rom_skeleton_view is not None:
                            self.rom_skeleton_view.update(first_kps, self.rom_movements, result_frame.shape)
                            grid_w = self.rom_skeleton_view.canvas_size[0]
                        else:
                            grid_w = self.angle_plotter.canvas_width if self.angle_plotter else 320

                        left_y = 0
                        if self.angle_plotter is not None and self.angle_plotter.joints:
                            jg = self.angle_plotter.render_grid('joint', grid_w)
                            if jg is not None:
                                result_frame = self.angle_plotter.attach_canvas(
                                    result_frame, jg, 'top_left', 10, left_y)
                                left_y += jg.shape[0] + 10
                        if self.angle_plotter is not None and self.angle_plotter.rom_movements:
                            rg = self.angle_plotter.render_grid('rom', grid_w)
                            if rg is not None:
                                result_frame = self.angle_plotter.attach_canvas(
                                    result_frame, rg, 'top_left', 10, left_y)
                                left_y += rg.shape[0] + 10
                        if self.rom_skeleton_view is not None:
                            result_frame = self.rom_skeleton_view.attach_to_frame(
                                result_frame, position='top_left', margin=10,
                                above_element_height=left_y)

                    # Store frame data
                    frame_data = {
                        'frame_id': metadata['frame_id'],
                        'timestamp': metadata['timestamp'],
                    }
                    
                    # Add tracking box if tracker is available
                    if self.tracker is not None and hasattr(self.tracker, 'last_known_bbox') and self.tracker.last_known_bbox is not None:
                        frame_data['track_box'] = self.tracker.last_known_bbox.tolist()
                    
                    # Add pose results if available
                    if self.pose_estimator is not None:
                        frame_data['detections'] = pose_results
                    
                    # Add face orientation results if available
                    if self.face_orientation is not None and len(face_orientation_results) > 0:
                        frame_data['face_orientation'] = face_orientation_results
                    
                    all_detection_data.append(frame_data)
                    
                    if out_writer:
                        out_writer.write(result_frame)

                    # Show output in real-time if enabled
                    if self.show_output:
                        display_frame = resize_frame_for_display(result_frame, self.screen_width, self.screen_height)
                        cv2.imshow('PhysioTrack - Full Inference', display_frame)
                        # Wait 1ms and check if user pressed 'q' to quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nUser interrupted processing (pressed 'q')")
                            break

                    if progress_callback:
                        progress_callback(metadata['frame_id'], self.total_frames, pose_results)
                
                # Clear batch
                frame_batch = []
                frame_batch_metadata = []
                
                # Real-time FPS monitoring
                if self.show_fps:
                    current_time = time.time()
                    if current_time - last_fps_print_time >= fps_print_interval:
                        elapsed = current_time - start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        # Build component-wise FPS dict
                        fps_dict = {"Pipeline": f"{current_fps:.2f}"}
                        
                        if len(self.detectors) > 0:
                            for idx, detector in enumerate(self.detectors):
                                if hasattr(detector, 'get_avg_fps'):
                                    det_fps = detector.get_avg_fps()
                                    fps_dict[f"Det[{idx}]"] = f"{det_fps:.2f}"
                        
                        if self.pose_estimator and hasattr(self.pose_estimator, 'pose_estimator'):
                            if hasattr(self.pose_estimator.pose_estimator, 'get_avg_fps'):
                                pose_fps = self.pose_estimator.pose_estimator.get_avg_fps()
                                fps_dict["Pose"] = f"{pose_fps:.2f}"
                        
                        if self.tracker and hasattr(self.tracker, 'get_avg_fps'):
                            tracker_fps = self.tracker.get_avg_fps()
                            fps_dict["Track"] = f"{tracker_fps:.2f}"
                        
                        if len(self.segmentators) > 0:
                            for idx, segmentor in enumerate(self.segmentators):
                                if hasattr(segmentor, 'get_avg_fps'):
                                    seg_fps = segmentor.get_avg_fps()
                                    fps_dict[f"Seg[{idx}]"] = f"{seg_fps:.2f}"
                        
                        if self.face_orientation and hasattr(self.face_orientation, 'get_avg_fps'):
                            face_fps = self.face_orientation.get_avg_fps()
                            fps_dict["Face"] = f"{face_fps:.2f}"

                        if self.depth_estimator and hasattr(self.depth_estimator, 'get_avg_fps'):
                            depth_fps = self.depth_estimator.get_avg_fps()
                            fps_dict["Depth"] = f"{depth_fps:.2f}"

                        fps_dict["Batch"] = f"{self.batch_size}"
                        
                        if pbar:
                            # Update progress bar with FPS info
                            pbar.set_postfix(fps_dict)
                        else:
                            # Print FPS on same line if no progress bar
                            fps_parts = [f"{k}: {v}" for k, v in fps_dict.items()]
                            fps_display = " | ".join(fps_parts)
                            print(f"\r{fps_display}", end='', flush=True)
                        
                        last_fps_print_time = current_time
            
            # Check if we're done
            if not ret and len(frame_batch) == 0:
                break
            
        if pbar:
            pbar.close()
            
        self.cap.release()
        if out_writer:
            out_writer.release()

        # Release ego video view if used
        if self.ego_view is not None:
            self.ego_view.release()

        # Close display window if show_output was enabled
        if self.show_output:
            cv2.destroyAllWindows()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        # Print detailed FPS statistics
        if self.show_fps:
            print("\n\n" + "="*60)  # Extra newline to clear real-time FPS line
            print("PERFORMANCE METRICS")
            print("="*60)
            print(f"Batch Size: {self.batch_size}")
            print(f"Overall Pipeline FPS: {avg_fps:.2f}")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f}s")
            print("-"*60)

            # Component-level FPS
            if len(self.detectors) > 0:
                for idx, detector in enumerate(self.detectors):
                    if hasattr(detector, 'get_avg_fps'):
                        det_fps = detector.get_avg_fps()
                        det_time = detector.get_avg_inference_time()
                        print(f"Detector[{idx}] FPS: {det_fps:.2f} ({det_time:.2f}ms per frame)")

            if self.pose_estimator and hasattr(self.pose_estimator, 'pose_estimator'):
                if hasattr(self.pose_estimator.pose_estimator, 'get_avg_fps'):
                    pose_fps = self.pose_estimator.pose_estimator.get_avg_fps()
                    pose_time = self.pose_estimator.pose_estimator.get_avg_inference_time()
                    print(f"Pose Estimator FPS: {pose_fps:.2f} ({pose_time:.2f}ms per frame)")

            if self.tracker and hasattr(self.tracker, 'get_avg_fps'):
                tracker_fps = self.tracker.get_avg_fps()
                tracker_time = self.tracker.get_avg_inference_time()
                print(f"Tracker FPS: {tracker_fps:.2f} ({tracker_time:.2f}ms per frame)")

            if len(self.segmentators) > 0:
                for idx, segmentor in enumerate(self.segmentators):
                    if hasattr(segmentor, 'get_avg_fps'):
                        seg_fps = segmentor.get_avg_fps()
                        seg_time = segmentor.get_avg_inference_time()
                        print(f"Segmentor[{idx}] FPS: {seg_fps:.2f} ({seg_time:.2f}ms per frame)")

            if self.face_detector and hasattr(self.face_detector, 'get_avg_fps'):
                face_det_fps = self.face_detector.get_avg_fps()
                face_det_time = self.face_detector.get_avg_inference_time()
                print(f"Face Detector FPS: {face_det_fps:.2f} ({face_det_time:.2f}ms per frame)")

            if self.face_orientation and hasattr(self.face_orientation, 'get_avg_fps'):
                face_orient_fps = self.face_orientation.get_avg_fps()
                face_orient_time = self.face_orientation.get_avg_inference_time()
                print(f"Face Orientation FPS: {face_orient_fps:.2f} ({face_orient_time:.2f}ms per frame)")

            if self.depth_estimator and hasattr(self.depth_estimator, 'get_avg_fps'):
                depth_fps = self.depth_estimator.get_avg_fps()
                depth_time = self.depth_estimator.get_avg_inference_time()
                print(f"Depth Estimator FPS: {depth_fps:.2f} ({depth_time:.2f}ms per frame)")

            print("="*60)
                  
        if output_json:
            self._save_json_data(all_detection_data, output_json)
        
        return all_detection_data
    
    def batch_run(self, 
                  input_paths: List[Union[str, Path]],
                  output_dir: Union[str, Path],
                  save_videos: bool = True,
                  save_json: bool = True):
        """Process several videos with this pipeline's configuration.

        Iterates the given paths, calling [`run`][physiotrack.Video.run] for each
        and writing ``<name>_processed.mp4`` / ``<name>_result.json`` into
        ``output_dir``.

        Args:
            input_paths (list[str | Path]): Video files to process.
            output_dir (str | Path): Directory for all outputs; created if missing.
            save_videos (bool): If ``True``, write an annotated MP4 per input.
                Defaults to ``True``.
            save_json (bool): If ``True``, write a JSON result file per input.
                Defaults to ``True``.

        Returns:
            dict[str, list[dict]]: Maps each input's file stem to its per-frame
                results (the return value of [`run`][physiotrack.Video.run]).

        Warning:
            The pipeline is configured for the single ``source`` passed to the
            constructor. This method reuses that same ``Video`` instance (and its
            already-opened capture) for every path, so it is best suited to reusing
            the model configuration rather than re-seeking multiple distinct clips.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for input_path in input_paths:
            input_path = Path(input_path)
            video_name = input_path.stem
            
            if self.verbose:
                print(f"Processing video: {input_path}")
            
            video_output_path = output_dir / f"{video_name}_processed.mp4" if save_videos else None
            json_output_path = output_dir / f"{video_name}_result.json" if save_json else None
            
            detection_data = self.run(video_output_path, json_output_path)
            results[video_name] = detection_data
            
        return results
    
    def _save_json_data(self, detection_data: List[Dict[str, Any]], output_path: Union[str, Path]):
        """Save detection data to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        if self.verbose:
            print(f"JSON data saved to: {output_path}")
