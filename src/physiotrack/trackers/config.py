import os 
from .._paths import weights_dir


class TrackerConfig:
    """Tunable configuration for a [`Tracker`][physiotrack.Tracker].

    Bundles every knob for the multi-object tracker: which backend to run, which
    detection classes to follow, per-backend hyper-parameters, the optional
    single-subject lock heuristic, and the on-frame overlay style.
    Every field has a sensible default, so ``TrackerConfig()`` is a valid,
    ready-to-use configuration.

    Settings can be provided two ways, and both may be mixed::

        cfg = TrackerConfig(tracker_type="ocsort", classes=[0])   # via constructor kwargs
        cfg.debug_mode = True                                # via attribute assignment

    Any keyword that is not a known field raises ``TypeError``, which guards against
    silent typos — including at construction time, so a misspelled setting is never
    accepted and then ignored.

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
        show_locked_subject (bool): Draw the blue box and ``Subject:<id>`` label for
            the locked subject (requires :attr:`enable_subject_lock`).
            Defaults to ``True``.
        show_tracking_tail (bool): Draw the blue movement trail for the locked
            subject. Defaults to ``True``.
        show_all_trails (bool): Draw movement trails for every track, not just the
            locked subject. Defaults to ``False``.
        tail_opacity (float): Opacity of the tracking tail overlay in ``[0.0, 1.0]``.
            Defaults to ``0.7``.
        debug_mode (bool): Log subject-lock diagnostics (track init/loss, IOU
            values) at ``DEBUG`` level. Defaults to ``False``.
        tracker_type (str): Which backend to use. One of ``"bytetrack"``,
            ``"strongsort"``, ``"ocsort"``, ``"boosttrack"`` (case-insensitive).
            Defaults to ``"ocsort"``.
        classes (list[int]): Detection class ids to keep and track; detections of
            other classes are dropped before tracking. Defaults to ``[0]`` (COCO
            "person").
        trail_length (int): Maximum number of points/boxes retained per track for
            trail drawing and history (``deque`` maxlen). Defaults to ``30``.
        enable_subject_lock (bool): Enable the single-subject lock heuristic, which
            follows one stable track and ignores the rest. Defaults to
            ``False``.
        required_consecutive_frames (int): Frames a track must persist to be
            promoted to the locked subject, and the miss count that drops it.
            Defaults to ``30``.
        inconsistent_motion_threshold (int): Number of consecutive low-IOU
            (inconsistent) frames tolerated before the lock is released.
            Defaults to ``5``.
        subject_reinit_iou_threshold (float): Minimum IOU in ``[0.0, 1.0]`` between
            the last known subject box and a candidate box to accept a match /
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

        # OC-SORT, tracking only the person class, with the subject overlay on.
        cfg = pt.TrackerConfig(
            tracker_type="ocsort",
            classes=[0],
            enable_subject_lock=True,
        )
        print(cfg)                  # inspect the resolved settings
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
        self.show_locked_subject = True     # Blue box for the locked subject
        self.show_tracking_tail = True     # Blue trail for the locked subject's movement
        self.show_all_trails = False       # Show trails for all tracks, not just the locked one
        self.tail_opacity = 0.7           # Opacity of tracking tail
        self.debug_mode = False
        
        # Tracker settings
        self.tracker_type = 'ocsort'  # Options: 'bytetrack', 'strongsort', 'ocsort', 'boosttrack'
        self.classes = [0]  # Classes to track
        self.trail_length = 30
        
        # Subject-lock settings
        self.enable_subject_lock = False
        self.required_consecutive_frames = 30
        self.inconsistent_motion_threshold = 5
        self.subject_reinit_iou_threshold = 0.3
        
        # ByteTrack settings
        self.bytetrack_track_thresh = 0.25
        self.bytetrack_match_thresh = 0.8
        self.bytetrack_track_buffer = 30
        self.bytetrack_frame_rate = 30
        
        # StrongSORT settings
        model_dir = str(weights_dir())
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

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f"Unknown TrackerConfig setting: {key!r}")
            setattr(self, key, value)

    def __str__(self):
        """Return the configuration grouped by category, for display.

        Implemented as ``__str__`` rather than a ``print()`` method so the object
        follows the usual Python convention — ``print(config)`` works, and the same text
        can be logged, written to a file, or embedded in a report instead of only going
        to stdout. A method named ``print`` also shadowed the builtin inside the class
        body.

        Returns:
            str: A multi-line summary. Only the settings relevant to the selected
                :attr:`tracker_type` are used at runtime, but all are listed.

        Example:
            ```python
            import physiotrack as pt

            config = pt.TrackerConfig(tracker_type="ocsort")
            print(config)
            ```
        """
        categories = {
            "General Settings": ["device", "text_size", "tracker_type", "classes", "trail_length"],
            "Subject Lock": ["enable_subject_lock", "required_consecutive_frames",
                                 "inconsistent_motion_threshold", "subject_reinit_iou_threshold"],
            "Overlay Options": ["show_detection_boxes", "show_original_tracks", "show_locked_subject",
                                "show_tracking_tail", "show_all_trails", "tail_opacity"],
            "ByteTrack": [k for k in vars(self) if k.startswith("bytetrack_")],
            "StrongSORT": [k for k in vars(self) if k.startswith("strongsort_")],
            "OCSort": [k for k in vars(self) if k.startswith("ocsort_")],
            "BoostTrack": [k for k in vars(self) if k.startswith("boosttrack_")],
        }

        rule = "=" * 60
        lines = [rule, " TRACKER CONFIGURATION ", rule]
        for category, keys in categories.items():
            lines.append(f"\n{category}:")
            for key in keys:
                if hasattr(self, key):
                    display_key = key.replace("_", " ").title()
                    lines.append(f"  {display_key:<35} : {getattr(self, key)}")
        lines.append(rule)
        return "\n".join(lines)