"""
Unified result objects for Physiotrack predictors.

Every image-based predictor (Detection, Pose, Segmentation, Face) returns a
``Result`` (or ``list[Result]`` for batches). Depth returns a ``DepthResult`` and
the tracker returns a ``TrackResult``. Each result carries the structured data plus
the source frame, and renders its own overlay via ``.plot()`` so that rendering is a
property of the *result*, not configured on the model.

See ``docs/API_REDESIGN.md`` for the full contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:  # cv2 is a hard dependency of the library, but keep results.py importable without it
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


__all__ = [
    "Keypoint",
    "Keypoints",
    "Instance",
    "Result",
    "DepthResult",
    "TrackResult",
]


# COCO-17 skeleton edges (used when drawing body keypoints).
_COCO17_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
]


def _name_maps(architecture: Optional[str]):
    """Lazily fetch keypoint id->name maps (avoids import-time circular imports)."""
    from .pose.config import COCO, COCO_WHOLEBODY
    return COCO_WHOLEBODY if architecture == "WHOLEBODY" else COCO


# --------------------------------------------------------------------------- #
# Keypoints
# --------------------------------------------------------------------------- #
class Keypoint:
    """A single body/face/hand landmark with pixel coordinates and confidence.

    A keypoint carries its integer id (indexing into the active skeleton, e.g.
    COCO-17 or COCO-WholeBody-133), its human-readable ``name``, its pixel
    location ``(x, y)`` in the source frame, an optional depth/root-relative
    ``z``, and a detection ``confidence``. Keypoints are produced by pose models
    and grouped inside a [`Keypoints`][physiotrack.Keypoints] collection on each
    [`Instance`][physiotrack.Instance].

    Attributes:
        id (int): Keypoint index in the active skeleton (e.g. ``0`` is
            ``"nose"`` for COCO).
        name (str): Human-readable joint name (e.g. ``"left_shoulder"``), or
            ``"unknown_<id>"`` if the id is not in the name map.
        x (float): X pixel coordinate in the source frame.
        y (float): Y pixel coordinate in the source frame.
        z (float | None): Depth or root-relative Z value when a 3D/depth-aware
            model produced it, otherwise ``None``.
        confidence (float): Detection confidence in ``[0.0, 1.0]``.

    Example:
        ```python
        import physiotrack as pt
        pose = pt.Pose.Person()
        result = pose.predict(frame)
        for kp in result.keypoints[0]:
            print(kp.name, kp.x, kp.y, kp.confidence)
        ```

    See Also:
        [`Keypoints`][physiotrack.Keypoints]: the ordered collection wrapper.
    """

    __slots__ = ("id", "name", "x", "y", "z", "confidence")

    def __init__(self, id: int, name: str, x: float, y: float,
                 confidence: float, z: Optional[float] = None):
        """Construct a keypoint.

        Args:
            id (int): Keypoint index in the active skeleton.
            name (str): Human-readable joint name.
            x (float): X pixel coordinate in the source frame.
            y (float): Y pixel coordinate in the source frame.
            confidence (float): Detection confidence in ``[0.0, 1.0]``.
            z (float, optional): Depth or root-relative Z value. Defaults to
                ``None`` (2D-only keypoint).
        """
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence

    def __repr__(self) -> str:
        z = "" if self.z is None else f", z={self.z:.1f}"
        return (f"Keypoint(id={self.id}, name='{self.name}', "
                f"x={self.x:.1f}, y={self.y:.1f}{z}, conf={self.confidence:.3f})")


class Keypoints:
    """Ordered collection of [`Keypoint`][physiotrack.Keypoint] for one subject.

    Wraps the per-instance landmarks and supports three lookup styles: positional
    (``kps[0]``), by skeleton id (``kps.by_id(5)``), and by name
    (``kps.by_name("left_shoulder")``). It also exposes vectorized NumPy views
    (:attr:`xy`, :attr:`xyz`, :attr:`conf`) for numeric work. Iteration and
    ``len()`` follow insertion order, which matches the model's skeleton order.

    Attributes:
        architecture (str): Skeleton the ids/names come from. ``"WHOLEBODY"``
            uses the COCO-WholeBody-133 name map; anything else uses COCO-17.

    Example:
        ```python
        import physiotrack as pt
        pose = pt.Pose.Person()
        kps = pose.predict(frame).keypoints[0]
        nose = kps.by_name("nose")
        print(len(kps), kps.xy.shape)        # e.g. 133 (133, 2)
        ```

    See Also:
        [`Keypoint`][physiotrack.Keypoint]: the individual landmark.
        [`Instance`][physiotrack.Instance]: holds one ``Keypoints`` as ``.keypoints``.
    """

    def __init__(self, keypoints_data: List[dict], architecture: str = "WHOLEBODY"):
        """Build a keypoint collection from raw model output.

        Args:
            keypoints_data (list[dict]): One dict per keypoint with keys
                ``"id"``, ``"x"``, ``"y"``, ``"confidence"``, and optionally
                ``"z"``. Ids are mapped to names via the architecture's name map.
            architecture (str, optional): Skeleton naming to apply. ``"WHOLEBODY"``
                uses COCO-WholeBody-133 names; any other value uses COCO-17.
                Defaults to ``"WHOLEBODY"``.
        """
        self.architecture = architecture
        names = _name_maps(architecture)
        self._ordered: List[Keypoint] = []
        self._by_id: Dict[int, Keypoint] = {}
        self._by_name: Dict[str, Keypoint] = {}

        for kp in keypoints_data:
            name = names.get(str(kp["id"]), f"unknown_{kp['id']}")
            keypoint = Keypoint(
                id=kp["id"], name=name, x=kp["x"], y=kp["y"],
                confidence=kp["confidence"], z=kp.get("z"),
            )
            self._ordered.append(keypoint)
            self._by_id[keypoint.id] = keypoint
            self._by_name[name] = keypoint

    # -- access -------------------------------------------------------------- #
    def by_id(self, keypoint_id: int) -> Optional[Keypoint]:
        """Look up a keypoint by its skeleton id.

        Args:
            keypoint_id (int): Skeleton index to fetch (e.g. ``0`` for nose).

        Returns:
            Keypoint | None: The matching [`Keypoint`][physiotrack.Keypoint], or
                ``None`` if no keypoint with that id is present.
        """
        return self._by_id.get(keypoint_id)

    def by_name(self, keypoint_name: str) -> Optional[Keypoint]:
        """Look up a keypoint by its joint name.

        Args:
            keypoint_name (str): Joint name to fetch (e.g. ``"left_shoulder"``).

        Returns:
            Keypoint | None: The matching [`Keypoint`][physiotrack.Keypoint], or
                ``None`` if no keypoint with that name is present.
        """
        return self._by_name.get(keypoint_name)

    def __getitem__(self, index: int) -> Keypoint:
        """Return the keypoint at the given positional index (skeleton order).

        Args:
            index (int): Zero-based position in skeleton order.

        Returns:
            Keypoint: The keypoint at ``index``.
        """
        return self._ordered[index]

    def __iter__(self):
        """Iterate over keypoints in skeleton order.

        Yields:
            Keypoint: Each [`Keypoint`][physiotrack.Keypoint] in order.
        """
        return iter(self._ordered)

    def __len__(self) -> int:
        """Return the number of keypoints in the collection.

        Returns:
            int: Count of keypoints (e.g. ``17`` for COCO, ``133`` for WholeBody).
        """
        return len(self._ordered)

    # -- vectorized views ---------------------------------------------------- #
    @property
    def xy(self) -> np.ndarray:
        """Pixel coordinates of every keypoint, in skeleton order.

        Returns:
            np.ndarray: Float32 array of shape ``(N, 2)`` holding ``(x, y)`` pixel
                coordinates for the ``N`` keypoints.
        """
        return np.array([[k.x, k.y] for k in self._ordered], dtype=np.float32)

    @property
    def xyz(self) -> Optional[np.ndarray]:
        """3D coordinates of every keypoint, when depth/Z is available.

        Returns:
            np.ndarray | None: Float32 array of shape ``(N, 3)`` holding
                ``(x, y, z)`` for the ``N`` keypoints, or ``None`` if the
                keypoints are 2D-only (first keypoint has no ``z``).
        """
        if not self._ordered or self._ordered[0].z is None:
            return None
        return np.array([[k.x, k.y, k.z] for k in self._ordered], dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        """Confidence of every keypoint, in skeleton order.

        Returns:
            np.ndarray: Float32 array of shape ``(N,)`` with per-keypoint
                confidences in ``[0.0, 1.0]``.
        """
        return np.array([k.confidence for k in self._ordered], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Instance (one detected subject)
# --------------------------------------------------------------------------- #
class Instance:
    """A single detected subject within a frame.

    ``Instance`` is the per-subject record inside a [`Result`][physiotrack.Result].
    Which fields are populated depends on the task: detection sets
    ``box``/``confidence``/``cls``/``cls_name``; pose adds ``keypoints``;
    segmentation may add a per-instance ``mask``; face orientation adds
    ``orientation``; tracking sets a persistent ``id``. Unused fields are ``None``.

    Attributes:
        id (int | None): Persistent track id (set by the tracker), otherwise
            ``None``.
        box (np.ndarray | None): Bounding box ``[x1, y1, x2, y2]`` in pixels,
            shape ``(4,)``, or ``None``.
        confidence (float | None): Detection confidence in ``[0.0, 1.0]``, or
            ``None``.
        cls (int | None): Integer class id, or ``None``.
        cls_name (str | None): Human-readable class name (e.g. ``"person"``), or
            ``None``.
        keypoints (Keypoints | None): Pose landmarks for this subject as a
            [`Keypoints`][physiotrack.Keypoints], or ``None``.
        mask (np.ndarray | None): Binary instance mask of shape ``(H, W)``, or
            ``None``.
        orientation (dict | None): Head pose dict ``{"yaw", "pitch", "roll"}``
            in degrees, or ``None``.

    Example:
        ```python
        import physiotrack as pt
        result = pt.Pose.Person().predict(frame)
        inst = result[0]
        print(inst.box, inst.confidence)
        if inst.keypoints is not None:
            print(inst.keypoints.by_name("nose"))
        ```

    See Also:
        [`Result`][physiotrack.Result]: the frame-level container of instances.
        [`Keypoints`][physiotrack.Keypoints]: the ``keypoints`` field type.
    """

    __slots__ = ("id", "box", "confidence", "cls", "cls_name",
                 "keypoints", "mask", "orientation")

    def __init__(self, *, id: Optional[int] = None,
                 box: Optional[np.ndarray] = None,
                 confidence: Optional[float] = None,
                 cls: Optional[int] = None,
                 cls_name: Optional[str] = None,
                 keypoints: Optional[Keypoints] = None,
                 mask: Optional[np.ndarray] = None,
                 orientation: Optional[dict] = None):
        """Construct an instance (all fields keyword-only and optional).

        Args:
            id (int, optional): Persistent track id. Defaults to ``None``.
            box (np.ndarray, optional): Bounding box ``[x1, y1, x2, y2]`` of
                shape ``(4,)``. Defaults to ``None``.
            confidence (float, optional): Detection confidence in ``[0.0, 1.0]``.
                Defaults to ``None``.
            cls (int, optional): Integer class id. Defaults to ``None``.
            cls_name (str, optional): Human-readable class name. Defaults to
                ``None``.
            keypoints (Keypoints, optional): Pose landmarks. Defaults to ``None``.
            mask (np.ndarray, optional): Binary instance mask of shape ``(H, W)``.
                Defaults to ``None``.
            orientation (dict, optional): Head pose ``{"yaw", "pitch", "roll"}``
                in degrees. Defaults to ``None``.
        """
        self.id = id
        self.box = box
        self.confidence = confidence
        self.cls = cls
        self.cls_name = cls_name
        self.keypoints = keypoints
        self.mask = mask
        self.orientation = orientation

    def __repr__(self) -> str:
        parts = [f"id={self.id}"]
        if self.box is not None:
            parts.append(f"box={np.round(np.asarray(self.box), 1).tolist()}")
        if self.confidence is not None:
            parts.append(f"conf={self.confidence:.3f}")
        if self.keypoints is not None:
            parts.append(f"keypoints={len(self.keypoints)}")
        if self.mask is not None:
            parts.append("mask=yes")
        if self.orientation is not None:
            parts.append(f"orientation={self.orientation}")
        return f"Instance({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# Result (detect / pose / segment / face)
# --------------------------------------------------------------------------- #
class Result:
    """Per-frame result shared by all image tasks (detect, pose, segment, face).

    Returned by every image-based predictor for a single frame (a
    ``list[Result]`` is returned for a batch). A ``Result`` bundles the source
    frame with the detected [`Instance`][physiotrack.Instance] objects and,
    for segmentation, a class-index map. It behaves like a sequence of instances
    (``len(result)``, ``result[i]``, iteration), exposes convenience views
    (:attr:`boxes`, :attr:`keypoints`), serializes via :meth:`to_dict`, and
    renders its own overlay via :meth:`plot` — so rendering is a property of the
    result, not configured on the model.

    Attributes:
        orig_img (np.ndarray): Source BGR frame ``(H, W, 3)`` the result was
            computed from.
        instances (list[Instance]): Detected subjects in the frame.
        task (str): Task that produced this result — one of ``"detect"``,
            ``"pose"``, ``"segment"``, ``"face"``.
        architecture (str | None): Model/skeleton hint (e.g. ``"WHOLEBODY"``) used
            when interpreting keypoints, or ``None``.
        seg_map (np.ndarray | None): Class-index map of shape ``(H, W)`` for
            segmentation tasks, otherwise ``None``.
        names (dict[int, str] | None): Class-id to class-name mapping, or ``None``.
        palette (np.ndarray | None): Optional ``(K, 3)`` RGB palette used to
            colorize ``seg_map`` (e.g. face parsing). When ``None`` the default
            segmentation palette is used.

    Example:
        ```python
        import physiotrack as pt
        det = pt.Detection.Person()
        result = det.predict(frame)
        print(len(result), result.boxes.shape)   # e.g. 3 (3, 4)
        annotated = result.plot(conf=True)
        data = result.to_dict()
        ```

    See Also:
        [`Instance`][physiotrack.Instance]: a single subject in the result.
        [`DepthResult`][physiotrack.DepthResult]: depth-task counterpart.
        [`TrackResult`][physiotrack.TrackResult]: tracker counterpart.
    """

    def __init__(self, *, orig_img: np.ndarray, instances: List[Instance],
                 task: str, architecture: Optional[str] = None,
                 seg_map: Optional[np.ndarray] = None,
                 names: Optional[Dict[int, str]] = None,
                 palette: Optional[np.ndarray] = None):
        """Construct a per-frame result (all fields keyword-only).

        Args:
            orig_img (np.ndarray): Source BGR frame of shape ``(H, W, 3)``.
            instances (list[Instance]): Detected subjects for this frame.
            task (str): Producing task: ``"detect"``, ``"pose"``, ``"segment"``,
                or ``"face"``.
            architecture (str, optional): Skeleton/model hint (e.g.
                ``"WHOLEBODY"``). Defaults to ``None``.
            seg_map (np.ndarray, optional): Class-index map ``(H, W)`` for
                segmentation. Defaults to ``None``.
            names (dict[int, str], optional): Class-id to name map. Defaults to
                ``None``.
            palette (np.ndarray, optional): ``(K, 3)`` RGB palette for colorizing
                ``seg_map``. Defaults to ``None`` (default palette).
        """
        self.orig_img = orig_img
        self.instances = instances
        self.task = task
        self.architecture = architecture
        self.seg_map = seg_map
        self.names = names
        # Optional (K, 3) RGB palette for colorizing ``seg_map`` (e.g. face parsing).
        # When None, the default segmentation palette is used.
        self.palette = palette

    # -- container protocol -------------------------------------------------- #
    def __iter__(self):
        """Iterate over the detected instances in the frame.

        Yields:
            Instance: Each [`Instance`][physiotrack.Instance] in ``instances``.
        """
        return iter(self.instances)

    def __len__(self) -> int:
        """Return the number of detected instances.

        Returns:
            int: Count of instances in the frame.
        """
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        """Return the instance at the given index.

        Args:
            index (int): Zero-based instance index.

        Returns:
            Instance: The [`Instance`][physiotrack.Instance] at ``index``.
        """
        return self.instances[index]

    def __repr__(self) -> str:
        return (f"Result(task='{self.task}', instances={len(self.instances)}"
                f"{', architecture=' + repr(self.architecture) if self.architecture else ''})")

    # -- convenience views --------------------------------------------------- #
    @property
    def boxes(self) -> np.ndarray:
        """Bounding boxes of all instances that have one.

        Returns:
            np.ndarray: Float32 array of shape ``(M, 4)`` with rows
                ``[x1, y1, x2, y2]`` for the ``M`` instances that have a box; an
                empty ``(0, 4)`` array when there are none.
        """
        boxes = [i.box for i in self.instances if i.box is not None]
        return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), np.float32)

    @property
    def keypoints(self) -> List[Keypoints]:
        """Keypoint collections for all instances that have pose landmarks.

        Returns:
            list[Keypoints]: One [`Keypoints`][physiotrack.Keypoints] per instance
                that has keypoints (instances without pose data are skipped).
        """
        return [i.keypoints for i in self.instances if i.keypoints is not None]

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain, JSON-friendly dict.

        Only populated instance fields are emitted. Boxes are written under the
        ``"bbox"`` key and orientation under ``"pose"``.

        Returns:
            dict: A dict with ``"task"`` (str) and ``"detections"`` (list of
                per-instance dicts). Each detection may contain ``"id"`` (int),
                ``"bbox"`` (``[x1, y1, x2, y2]``), ``"confidence"`` (float),
                ``"cls"`` (int), ``"keypoints"`` (list of ``{"id", "x", "y",
                "confidence", "z"?}``), and ``"pose"`` (``{"yaw", "pitch",
                "roll"}``). The key ``"architecture"`` is added when set.

        Example:
            ```python
            import physiotrack as pt
            data = pt.Pose.Person().predict(frame).to_dict()
            data["task"], len(data["detections"])
            ```
        """
        detections = []
        for inst in self.instances:
            det: Dict[str, Any] = {}
            if inst.id is not None:
                det["id"] = inst.id
            if inst.box is not None:
                det["bbox"] = np.asarray(inst.box).tolist()
            if inst.confidence is not None:
                det["confidence"] = float(inst.confidence)
            if inst.cls is not None:
                det["cls"] = int(inst.cls)
            if inst.keypoints is not None:
                det["keypoints"] = [
                    {"id": k.id, "x": k.x, "y": k.y, "confidence": k.confidence,
                     **({"z": k.z} if k.z is not None else {})}
                    for k in inst.keypoints
                ]
            if inst.orientation is not None:
                det["pose"] = inst.orientation
            detections.append(det)
        out: Dict[str, Any] = {"task": self.task, "detections": detections}
        if self.architecture is not None:
            out["architecture"] = self.architecture
        return out

    # -- rendering ----------------------------------------------------------- #
    def plot(self, *, boxes: bool = True, labels: bool = True,
             keypoints: bool = True, masks: bool = True, conf: bool = False,
             color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Render an annotated copy of the source frame.

        Drawing is controlled here rather than on the model, so the same result can
        be drawn in different ways without re-running inference. Boxes, class/id
        labels, pose skeletons, segmentation masks, and head-pose axes are drawn
        based on what each instance carries and the toggles below. The original
        frame is not modified.

        Args:
            boxes (bool, optional): Draw bounding boxes. Defaults to ``True``.
            labels (bool, optional): Draw class/track-id labels above boxes (only
                when ``boxes`` is also drawn). Defaults to ``True``.
            keypoints (bool, optional): Draw pose keypoints and the COCO-17
                skeleton. Defaults to ``True``.
            masks (bool, optional): Blend segmentation masks (only for the
                ``"segment"`` task). Defaults to ``True``.
            conf (bool, optional): Append the detection confidence to labels.
                Defaults to ``False``.
            color (tuple, optional): Box/label BGR color. Defaults to
                ``(0, 255, 0)`` (green).
            thickness (int, optional): Box line thickness in pixels. Defaults to
                ``2``.

        Returns:
            np.ndarray: A new annotated BGR image of shape ``(H, W, 3)``.

        Raises:
            RuntimeError: If OpenCV (``cv2``) is not installed.

        Example:
            ```python
            import physiotrack as pt
            result = pt.Detection.Person().predict(frame)
            annotated = result.plot(conf=True, color=(0, 0, 255))
            ```

        Note:
            Head-pose axes are always drawn for instances that carry an
            ``orientation``, regardless of the toggles above.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for Result.plot().")
        img = self.orig_img.copy()

        if masks and self.task == "segment":
            img = self._draw_masks(img)

        for inst in self.instances:
            if boxes and inst.box is not None:
                x1, y1, x2, y2 = [int(v) for v in inst.box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                if labels:
                    self._draw_label(img, inst, x1, y1, conf, color)
            if keypoints and inst.keypoints is not None:
                self._draw_keypoints(img, inst.keypoints)
            if inst.orientation is not None:
                self._draw_orientation(img, inst)

        return img

    # -- drawing helpers ----------------------------------------------------- #
    @staticmethod
    def _draw_label(img, inst, x1, y1, show_conf, color):
        label = inst.cls_name if inst.cls_name else (
            f"id {inst.id}" if inst.id is not None else inst.task if False else "")
        if inst.id is not None and inst.cls_name:
            label = f"{inst.cls_name} {inst.id}"
        if show_conf and inst.confidence is not None:
            label = f"{label} {inst.confidence:.2f}".strip()
        if not label:
            return
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_keypoints(img, keypoints: "Keypoints", conf_thresh: float = 0.3):
        # skeleton (body-17 only; higher ids drawn as points)
        for a, b in _COCO17_SKELETON:
            ka, kb = keypoints.by_id(a), keypoints.by_id(b)
            if ka and kb and ka.confidence > conf_thresh and kb.confidence > conf_thresh:
                cv2.line(img, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)),
                         (255, 128, 0), 2, cv2.LINE_AA)
        for kp in keypoints:
            if kp.confidence > conf_thresh:
                cv2.circle(img, (int(kp.x), int(kp.y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    def _draw_masks(self, img):
        # Class-index map (the common segmentation output): colorize and blend.
        if self.seg_map is not None:
            try:
                if self.palette is not None:
                    # Palette-based colorizing (e.g. SegFace face parsing). Blend only
                    # over foreground (class > 0) so the background frame is untouched.
                    idx = np.clip(self.seg_map, 0, len(self.palette) - 1)
                    color_map = cv2.cvtColor(self.palette[idx].astype(np.uint8),
                                             cv2.COLOR_RGB2BGR)
                    if color_map.shape[:2] != img.shape[:2]:
                        color_map = cv2.resize(color_map, (img.shape[1], img.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                    fg = (self.seg_map > 0)
                    if fg.shape[:2] != img.shape[:2]:
                        fg = cv2.resize(fg.astype(np.uint8), (img.shape[1], img.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    blended = cv2.addWeighted(color_map, 0.5, img, 0.5, 0)
                    img = np.where(fg[..., None], blended, img)
                else:
                    from .modules import draw_segmentation_map
                    color_map = draw_segmentation_map(self.seg_map)
                    if color_map.shape[:2] != img.shape[:2]:
                        color_map = cv2.resize(color_map, (img.shape[1], img.shape[0]))
                    img = cv2.addWeighted(color_map, 0.5, img, 0.5, 0)
            except Exception:
                pass
        # Per-instance binary masks (if a backend provides them).
        rng = np.random.default_rng(0)
        for inst in self.instances:
            if inst.mask is None:
                continue
            mask = inst.mask.astype(bool)
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(inst.mask.astype(np.uint8),
                                  (img.shape[1], img.shape[0])).astype(bool)
            tint = rng.integers(64, 255, size=3).tolist()
            overlay = img.copy()
            overlay[mask] = tint
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        return img

    @staticmethod
    def _draw_orientation(img, inst):
        from .modules._6DRepNet360.utils import draw_axis
        o = inst.orientation or {}
        if not all(k in o for k in ("yaw", "pitch", "roll")):
            return
        if inst.box is not None:
            x1, y1, x2, y2 = [int(v) for v in inst.box]
            tdx, tdy = (x1 + x2) // 2, (y1 + y2) // 2
        else:
            tdx = tdy = None
        draw_axis(img, o["yaw"], o["pitch"], o["roll"], tdx=tdx, tdy=tdy)


# --------------------------------------------------------------------------- #
# DepthResult
# --------------------------------------------------------------------------- #
class DepthResult:
    """Dense depth result: the raw depth map plus colorization via :meth:`plot`.

    Returned by depth predictors for a single frame. Holds the source frame and a
    per-pixel depth map. Use :meth:`normalized` for a ``[0, 1]`` map or
    :meth:`plot` for a colorized BGR image ready to display.

    Attributes:
        orig_img (np.ndarray): Source BGR frame ``(H, W, 3)``.
        depth (np.ndarray): Raw per-pixel depth map of shape ``(H, W)``; larger
            values are nearer or farther depending on the backend.

    Example:
        ```python
        import physiotrack as pt
        d = pt.Depth.DepthAnythingV2Base().predict(frame)
        raw = d.depth                       # (H, W) float depth
        colored = d.plot(colormap="viridis")
        ```

    See Also:
        [`Result`][physiotrack.Result]: image-task counterpart.
    """

    _COLORMAPS = {
        "inferno": "COLORMAP_INFERNO", "viridis": "COLORMAP_VIRIDIS",
        "magma": "COLORMAP_MAGMA", "plasma": "COLORMAP_PLASMA", "jet": "COLORMAP_JET",
    }

    def __init__(self, *, orig_img: np.ndarray, depth: np.ndarray):
        """Construct a depth result (fields keyword-only).

        Args:
            orig_img (np.ndarray): Source BGR frame of shape ``(H, W, 3)``.
            depth (np.ndarray): Raw depth map of shape ``(H, W)``.
        """
        self.orig_img = orig_img
        self.depth = depth

    def __repr__(self) -> str:
        return f"DepthResult(shape={tuple(self.depth.shape)})"

    def normalized(self) -> np.ndarray:
        """Return the depth map min-max normalized to ``[0, 1]``.

        Returns:
            np.ndarray: Float32 map of shape ``(H, W)`` scaled to ``[0.0, 1.0]``.
                Returns an all-zeros map when the depth is (near-)constant.
        """
        d = self.depth.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-8:
            return np.zeros_like(d)
        return (d - lo) / (hi - lo)

    def plot(self, *, colormap: str = "inferno") -> np.ndarray:
        """Colorize the depth map into a displayable BGR image.

        The depth is min-max normalized (see :meth:`normalized`) and mapped
        through an OpenCV colormap.

        Args:
            colormap (str, optional): Colormap name. One of ``"inferno"``,
                ``"viridis"``, ``"magma"``, ``"plasma"``, ``"jet"``. Unknown
                names fall back to ``"inferno"``. Defaults to ``"inferno"``.

        Returns:
            np.ndarray: Colorized BGR image of shape ``(H, W, 3)``, dtype uint8.

        Raises:
            RuntimeError: If OpenCV (``cv2``) is not installed.

        Example:
            ```python
            import physiotrack as pt
            colored = pt.Depth.DepthAnythingV2Base().predict(frame).plot(colormap="magma")
            ```
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for DepthResult.plot().")
        norm = (self.normalized() * 255).astype(np.uint8)
        cmap = getattr(cv2, self._COLORMAPS.get(colormap, "COLORMAP_INFERNO"))
        return cv2.applyColorMap(norm, cmap)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize a lightweight summary of the depth result.

        Returns:
            dict: ``{"task": "depth", "shape": [H, W]}``. The raw depth array is
                not included.
        """
        return {"task": "depth", "shape": list(self.depth.shape)}


# --------------------------------------------------------------------------- #
# TrackResult
# --------------------------------------------------------------------------- #
class TrackResult:
    """Multi-object tracker output: instances each carrying a persistent ``id``.

    Returned per frame by the tracker. Behaves like a sequence of tracked
    [`Instance`][physiotrack.Instance] objects (``len``, indexing, iteration) and
    exposes the active track :attr:`ids` and :attr:`boxes`. It may also carry the
    tracker's own rich overlay in ``rendered`` (used by :meth:`plot` when no frame
    is supplied) and the backend's ``raw`` target rows.

    Attributes:
        instances (list[Instance]): Tracked subjects; each has a persistent
            ``id`` and usually a ``box``.
        orig_img (np.ndarray | None): Source BGR frame ``(H, W, 3)``, or ``None``.
        rendered (np.ndarray | None): The tracker's own pre-rendered overlay
            image (boxes, trails, etc.), or ``None``.
        raw (list): Backend raw target rows, each
            ``[x1, y1, x2, y2, id, (cls), (conf)]``. Empty list when unset.

    Example:
        ```python
        import numpy as np
        import physiotrack as pt
        det = pt.Detection.Person()
        tracker = pt.Tracker(pt.TrackerConfig(tracker="ocsort", classes=[0]))
        res = det.predict(frame)
        # Tracker expects an (N, 6) [x1, y1, x2, y2, conf, cls] array:
        detections = np.array([[*i.box, i.confidence, i.cls] for i in res],
                              dtype=np.float32) if len(res) else np.empty((0, 6), np.float32)
        track_result = tracker.track(frame, detections)
        print(track_result.ids)              # e.g. [1, 2, 5]
        annotated = track_result.plot()      # tracker's own overlay
        ```

    See Also:
        [`Result`][physiotrack.Result]: image-task counterpart.
        [`Instance`][physiotrack.Instance]: a single tracked subject.
    """

    def __init__(self, *, instances: List[Instance],
                 orig_img: Optional[np.ndarray] = None,
                 rendered: Optional[np.ndarray] = None,
                 raw: Optional[list] = None):
        """Construct a tracker result (fields keyword-only).

        Args:
            instances (list[Instance]): Tracked subjects for this frame.
            orig_img (np.ndarray, optional): Source BGR frame ``(H, W, 3)``.
                Defaults to ``None``.
            rendered (np.ndarray, optional): Tracker's own overlay image.
                Defaults to ``None``.
            raw (list, optional): Backend raw target rows
                ``[x1, y1, x2, y2, id, (cls), (conf)]``. Defaults to ``None``
                (stored as an empty list).
        """
        self.instances = instances
        self.orig_img = orig_img
        # ``rendered`` is the tracker's own rich overlay (student box, trails, etc.).
        self.rendered = rendered
        # ``raw`` is the backend's raw target rows: [x1,y1,x2,y2,id,(cls),(conf)].
        self.raw = raw if raw is not None else []

    def __iter__(self):
        """Iterate over the tracked instances.

        Yields:
            Instance: Each tracked [`Instance`][physiotrack.Instance].
        """
        return iter(self.instances)

    def __len__(self) -> int:
        """Return the number of active tracks.

        Returns:
            int: Count of tracked instances.
        """
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        """Return the tracked instance at the given index.

        Args:
            index (int): Zero-based track index.

        Returns:
            Instance: The [`Instance`][physiotrack.Instance] at ``index``.
        """
        return self.instances[index]

    def __repr__(self) -> str:
        return f"TrackResult(tracks={len(self.instances)})"

    @property
    def ids(self) -> List[int]:
        """Persistent track ids of all instances that have one.

        Returns:
            list[int]: Track ids, in instance order (instances without an id are
                skipped).
        """
        return [i.id for i in self.instances if i.id is not None]

    @property
    def boxes(self) -> np.ndarray:
        """Bounding boxes of all tracked instances that have one.

        Returns:
            np.ndarray: Float32 array of shape ``(M, 4)`` with rows
                ``[x1, y1, x2, y2]``; an empty ``(0, 4)`` array when there are
                none.
        """
        boxes = [i.box for i in self.instances if i.box is not None]
        return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), np.float32)

    def plot(self, frame: np.ndarray = None, *, boxes: bool = True, labels: bool = True,
             color: tuple = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """Render tracked boxes and ids onto a frame.

        If ``frame`` is omitted, returns the tracker's own ``rendered`` overlay
        when available, else draws on ``orig_img``. When a ``frame`` is given, a
        copy is annotated and returned (the input is not modified).

        Args:
            frame (np.ndarray, optional): BGR frame ``(H, W, 3)`` to draw on.
                Defaults to ``None`` (use ``rendered`` or ``orig_img``).
            boxes (bool, optional): Draw bounding boxes. Defaults to ``True``.
            labels (bool, optional): Draw ``"ID <n>"`` labels above boxes.
                Defaults to ``True``.
            color (tuple, optional): Box/label BGR color. Defaults to
                ``(255, 0, 0)`` (blue).
            thickness (int, optional): Box line thickness in pixels. Defaults to
                ``2``.

        Returns:
            np.ndarray: Annotated BGR image of shape ``(H, W, 3)``.

        Raises:
            ValueError: If no ``frame`` is supplied and neither ``rendered`` nor
                ``orig_img`` is available.
            RuntimeError: If OpenCV (``cv2``) is not installed.

        Example:
            ```python
            import numpy as np
            import physiotrack as pt
            det = pt.Detection.Person()
            tracker = pt.Tracker(pt.TrackerConfig(tracker="ocsort", classes=[0]))
            res = det.predict(frame)
            detections = np.array([[*i.box, i.confidence, i.cls] for i in res],
                                  dtype=np.float32) if len(res) else np.empty((0, 6), np.float32)
            tr = tracker.track(frame, detections)
            annotated = tr.plot(frame, color=(0, 255, 0))
            ```
        """
        # With no frame, return the tracker's own rich overlay if available.
        if frame is None:
            if self.rendered is not None:
                return self.rendered
            if self.orig_img is None:
                raise ValueError("No frame supplied and no rendered/orig_img available.")
            frame = self.orig_img
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for TrackResult.plot().")
        img = frame.copy()
        for inst in self.instances:
            if boxes and inst.box is not None:
                x1, y1, x2, y2 = [int(v) for v in inst.box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                if labels and inst.id is not None:
                    cv2.putText(img, f"ID {inst.id}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tracker result to a plain, JSON-friendly dict.

        Returns:
            dict: ``{"task": "track", "tracks": [...]}`` where each track is
                ``{"id": int, "bbox": [x1, y1, x2, y2] | None,
                "confidence": float | None, "cls": int | None}``.
        """
        return {
            "task": "track",
            "tracks": [
                {"id": i.id,
                 "bbox": (np.asarray(i.box).tolist() if i.box is not None else None),
                 "confidence": i.confidence, "cls": i.cls}
                for i in self.instances
            ],
        }
