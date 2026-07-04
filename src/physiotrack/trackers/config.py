import os 


class TrackerConfig:
    """Tunable configuration for a [`Tracker`][physiotrack.Tracker].

    Bundles every knob for the multi-object tracker: which backend to run, which
    detection classes to follow, per-backend hyper-parameters, the optional
    "student" (single-subject) tracking heuristic, and the on-frame overlay style.
    Every field has a sensible default, so ``TrackerConfig()`` is a valid,
    ready-to-use configuration.

    Settings can be provided two ways, and both may be mixed::

        cfg = TrackerConfig(tracker="ocsort", classes=[0])   # via constructor kwargs
        cfg.debug_mode = True                                # via attribute assignment

    ``tracker=`` is a friendly alias for :attr:`tracker_type`. Any keyword that is
    not a known field raises ``TypeError``, which guards against silent typos.

    Only the fields relevant to the selected :attr:`tracker_type` are used at
    runtime; the other backends' hyper-parameters are ignored but remain settable.

    Attributes:
        device (str): Torch device for backends that run a neural model (currently
            only StrongSORT's ReID network). ``"cuda"`` or ``"cpu"``. Defaults to
            ``"cuda"``.
        colors (dict[str, tuple[int, int, int]]): Named BGR colors used by the
            overlay. Keys ``"blue"``, ``"red"``, ``"green"``, ``"yellow"``,
            ``"purple"``. Defaults to a fixed palette (e.g. ``"blue"`` is
            ``(255, 0, 0)`` in BGR).
        text_size (int): Base text scale passed to the overlay label drawing.
            Defaults to ``1``.
        show_detection_boxes (bool): Draw red boxes for the raw (pre-tracking)
            detections. Defaults to ``False``.
        show_original_tracks (bool): Draw green boxes and ``ID:<n>`` labels for
            every multi-object track. Defaults to ``False``.
        show_student_track (bool): Draw the blue box and ``Student:<id>`` label for
            the isolated student track (requires :attr:`enable_student_tracking`).
            Defaults to ``True``.
        show_tracking_tail (bool): Draw the blue movement trail for the student
            track. Defaults to ``True``.
        show_all_trails (bool): Draw movement trails for every track, not just the
            student. Defaults to ``False``.
        tail_opacity (float): Opacity of the tracking tail overlay in ``[0.0, 1.0]``.
            Defaults to ``0.7``.
        debug_mode (bool): Print verbose student-tracking diagnostics (track
            init/loss, IOU values). Defaults to ``False``.
        tracker_type (str): Which backend to use. One of ``"bytetrack"``,
            ``"strongsort"``, ``"ocsort"``, ``"boosttrack"`` (case-insensitive).
            Defaults to ``"ocsort"``. Alias: pass ``tracker=`` to the constructor.
        classes (list[int]): Detection class ids to keep and track; detections of
            other classes are dropped before tracking. Defaults to ``[0]`` (COCO
            "person").
        trail_length (int): Maximum number of points/boxes retained per track for
            trail drawing and history (``deque`` maxlen). Defaults to ``30``.
        enable_student_tracking (bool): Enable the single-subject ("student")
            isolation heuristic that locks onto one stable track. Defaults to
            ``False``.
        required_consecutive_frames (int): Frames a track must persist to be
            promoted to the student track, and the miss count that drops it.
            Defaults to ``30``.
        inconsistent_motion_threshold (int): Number of consecutive low-IOU
            (inconsistent) frames tolerated before the student track is discarded.
            Defaults to ``5``.
        student_reinit_iou_threshold (float): Minimum IOU in ``[0.0, 1.0]`` between
            the last known student box and a candidate box to accept a match /
            re-initialization. Defaults to ``0.3``.
        bytetrack_track_thresh (float): ByteTrack detection-confidence threshold for
            the high-score association step, in ``[0.0, 1.0]``. Defaults to ``0.25``.
        bytetrack_match_thresh (float): ByteTrack IOU matching threshold in
            ``[0.0, 1.0]``. Defaults to ``0.8``.
        bytetrack_track_buffer (int): Frames a lost ByteTrack track is kept before
            removal. Defaults to ``30``.
        bytetrack_frame_rate (int): Assumed video frame rate (fps) used to scale the
            ByteTrack buffer. Defaults to ``30``.
        strongsort_reid_weights (str): Filesystem path to the StrongSORT ReID model
            weights. Defaults to ``osnet_x0_25_msmt17.pt`` under the package
            ``model_data`` directory.
        strongsort_max_dist (float): Maximum cosine appearance distance for a valid
            ReID match in ``[0.0, 1.0]``. Defaults to ``0.2``.
        strongsort_max_iou_dist (float): Maximum IOU distance (``1 - IOU``) for the
            motion association gate in ``[0.0, 1.0]``. Defaults to ``0.7``.
        strongsort_max_age (int): Frames a track survives without a match before
            deletion. Defaults to ``70``.
        strongsort_max_unmatched_preds (int): Maximum consecutive predicted-only
            (unmatched) updates allowed. Defaults to ``7``.
        strongsort_n_init (int): Consecutive detections required to confirm a new
            track. Defaults to ``3``.
        strongsort_nn_budget (int): Maximum number of appearance samples stored per
            track for the nearest-neighbour metric. Defaults to ``100``.
        strongsort_mc_lambda (float): Weight balancing motion vs. appearance in the
            matching cost, in ``[0.0, 1.0]``. Defaults to ``0.995``.
        strongsort_ema_alpha (float): EMA momentum for updating a track's appearance
            feature, in ``[0.0, 1.0]``. Defaults to ``0.9``.
        ocsort_det_thresh (float): OC-SORT detection-confidence threshold in
            ``[0.0, 1.0]``. Defaults to ``0.2``.
        ocsort_max_age (int): Frames an OC-SORT track survives without a match before
            deletion. Defaults to ``30``.
        ocsort_min_hits (int): Detections required before a track is reported.
            Defaults to ``3``.
        ocsort_iou_thresh (float): IOU threshold for OC-SORT association in
            ``[0.0, 1.0]``. Defaults to ``0.3``.
        ocsort_delta_t (int): Time gap (frames) used for OC-SORT's observation-centric
            velocity direction estimate. Defaults to ``3``.
        ocsort_asso_func (str): OC-SORT association metric, e.g. ``"iou"``. Defaults
            to ``"iou"``.
        ocsort_inertia (float): Weight of the velocity-direction consistency term in
            ``[0.0, 1.0]``. Defaults to ``0.2``.
        ocsort_use_byte (bool): Enable OC-SORT's ByteTrack-style low-score
            second association pass. Defaults to ``False``.
        boosttrack_det_thresh (float): BoostTrack detection-confidence threshold in
            ``[0.0, 1.0]``. Defaults to ``0.2``.
        boosttrack_lambda_iou (float): Weight of the IOU term in the BoostTrack
            similarity. Defaults to ``0.5``.
        boosttrack_lambda_mhd (float): Weight of the Mahalanobis-distance term.
            Defaults to ``0.5``.
        boosttrack_lambda_shape (float): Weight of the box-shape term. Defaults to
            ``0.5``.
        boosttrack_dlo_boost_coef (float): Detection-Likelihood-of-Object boost
            coefficient in ``[0.0, 1.0]``. Defaults to ``0.9``.
        boosttrack_use_dlo_boost (bool): Enable the detection-confidence (DLO) boost.
            Defaults to ``True``.
        boosttrack_use_duo_boost (bool): Enable the detection-uncertainty (DUO)
            boost. Defaults to ``True``.
        boosttrack_max_age (int): Frames a BoostTrack track survives without a match
            before deletion. Defaults to ``30``.

    Raises:
        TypeError: If a constructor keyword is not a recognized configuration field.

    Example:
        ```python
        import physiotrack as pt

        # OC-SORT, tracking only the person class, with the student overlay on.
        cfg = pt.TrackerConfig(
            tracker="ocsort",
            classes=[0],
            enable_student_tracking=True,
        )
        cfg.print()                 # inspect the resolved settings
        tracker = pt.Tracker(cfg)
        ```

    See Also:
        [`Tracker`][physiotrack.Tracker]: the tracker that consumes this config.
    """
    def __init__(self, **kwargs):
        # General settings
        self.device = 'cuda'
        self.colors = {
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255)
        }
        self.text_size = 1
        
        # Overlay options
        self.show_detection_boxes = False  # Red boxes for raw detections
        self.show_original_tracks = False   # Green boxes for all MOT tracks
        self.show_student_track = True     # Blue box for isolated student track
        self.show_tracking_tail = True     # Blue trail for student movement
        self.show_all_trails = False       # Show trails for all tracks (not just student)
        self.tail_opacity = 0.7           # Opacity of tracking tail
        self.debug_mode = False
        
        # Tracker settings
        self.tracker_type = 'ocsort'  # Options: 'bytetrack', 'strongsort', 'ocsort', 'boosttrack'
        self.classes = [0]  # Classes to track
        self.trail_length = 30
        
        # Student tracking settings
        self.enable_student_tracking = False
        self.required_consecutive_frames = 30
        self.inconsistent_motion_threshold = 5
        self.student_reinit_iou_threshold = 0.3
        
        # ByteTrack settings
        self.bytetrack_track_thresh = 0.25
        self.bytetrack_match_thresh = 0.8
        self.bytetrack_track_buffer = 30
        self.bytetrack_frame_rate = 30
        
        # StrongSORT settings
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        self.strongsort_reid_weights = os.path.join(model_dir, 'osnet_x0_25_msmt17.pt')
        self.strongsort_max_dist = 0.2
        self.strongsort_max_iou_dist = 0.7
        self.strongsort_max_age = 70
        self.strongsort_max_unmatched_preds = 7
        self.strongsort_n_init = 3
        self.strongsort_nn_budget = 100
        self.strongsort_mc_lambda = 0.995
        self.strongsort_ema_alpha = 0.9
        
        # OCSort settings
        self.ocsort_det_thresh = 0.2
        self.ocsort_max_age = 30
        self.ocsort_min_hits = 3
        self.ocsort_iou_thresh = 0.3
        self.ocsort_delta_t = 3
        self.ocsort_asso_func = 'iou'
        self.ocsort_inertia = 0.2
        self.ocsort_use_byte = False
        
        # BoostTrack settings
        self.boosttrack_det_thresh = 0.2
        self.boosttrack_lambda_iou = 0.5
        self.boosttrack_lambda_mhd = 0.5
        self.boosttrack_lambda_shape = 0.5
        self.boosttrack_dlo_boost_coef = 0.9
        self.boosttrack_use_dlo_boost = True
        self.boosttrack_use_duo_boost = True
        self.boosttrack_max_age = 30

        # Apply keyword overrides. `tracker` is an alias for `tracker_type`.
        if 'tracker' in kwargs:
            kwargs['tracker_type'] = kwargs.pop('tracker')
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f"Unknown TrackerConfig setting: {key!r}")
            setattr(self, key, value)

    def print(self):
        """Print all configuration settings in an organized format."""
        print("\n" + "="*60)
        print(" TRACKER CONFIGURATION ")
        print("="*60)
        
        # Group configs by category
        categories = {
            "General Settings": ["device", "text_size", "tracker_type", "classes", "trail_length"],
            "Student Tracking": ["enable_student_tracking", "required_consecutive_frames", 
                               "inconsistent_motion_threshold", "student_reinit_iou_threshold"],
            "Overlay Options": ["show_detection_boxes", "show_original_tracks", "show_student_track",
                              "show_tracking_tail", "show_all_trails", "tail_opacity"],
            "ByteTrack": [k for k in vars(self) if k.startswith("bytetrack_")],
            "StrongSORT": [k for k in vars(self) if k.startswith("strongsort_")],
            "OCSort": [k for k in vars(self) if k.startswith("ocsort_")],
            "BoostTrack": [k for k in vars(self) if k.startswith("boosttrack_")]
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    # Format the key name for display
                    display_key = key.replace("_", " ").title()
                    print(f"  {display_key:<35} : {value}")
        
        print("\n" + "="*60 + "\n")