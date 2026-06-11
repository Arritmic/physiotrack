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
    """A single keypoint with pixel coordinates and (optionally) a depth value."""

    __slots__ = ("id", "name", "x", "y", "z", "confidence")

    def __init__(self, id: int, name: str, x: float, y: float,
                 confidence: float, z: Optional[float] = None):
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
    """Ordered collection of :class:`Keypoint`, addressable by index, id, or name."""

    def __init__(self, keypoints_data: List[dict], architecture: str = "WHOLEBODY"):
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
        return self._by_id.get(keypoint_id)

    def by_name(self, keypoint_name: str) -> Optional[Keypoint]:
        return self._by_name.get(keypoint_name)

    def __getitem__(self, index: int) -> Keypoint:
        return self._ordered[index]

    def __iter__(self):
        return iter(self._ordered)

    def __len__(self) -> int:
        return len(self._ordered)

    # -- vectorized views ---------------------------------------------------- #
    @property
    def xy(self) -> np.ndarray:
        return np.array([[k.x, k.y] for k in self._ordered], dtype=np.float32)

    @property
    def xyz(self) -> Optional[np.ndarray]:
        if not self._ordered or self._ordered[0].z is None:
            return None
        return np.array([[k.x, k.y, k.z] for k in self._ordered], dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        return np.array([k.confidence for k in self._ordered], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Instance (one detected subject)
# --------------------------------------------------------------------------- #
class Instance:
    """A single detected subject within a frame.

    Fields are populated according to the task: a detection has ``box``/``confidence``/
    ``cls``; a pose instance adds ``keypoints``; a segmentation instance adds ``mask``;
    a face-orientation instance adds ``orientation``.
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
    """Per-frame result shared by all image tasks (detect, pose, segment, face)."""

    def __init__(self, *, orig_img: np.ndarray, instances: List[Instance],
                 task: str, architecture: Optional[str] = None,
                 seg_map: Optional[np.ndarray] = None,
                 names: Optional[Dict[int, str]] = None):
        self.orig_img = orig_img
        self.instances = instances
        self.task = task
        self.architecture = architecture
        self.seg_map = seg_map
        self.names = names

    # -- container protocol -------------------------------------------------- #
    def __iter__(self):
        return iter(self.instances)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]

    def __repr__(self) -> str:
        return (f"Result(task='{self.task}', instances={len(self.instances)}"
                f"{', architecture=' + repr(self.architecture) if self.architecture else ''})")

    # -- convenience views --------------------------------------------------- #
    @property
    def boxes(self) -> np.ndarray:
        boxes = [i.box for i in self.instances if i.box is not None]
        return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), np.float32)

    @property
    def keypoints(self) -> List[Keypoints]:
        return [i.keypoints for i in self.instances if i.keypoints is not None]

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
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
        be drawn in different ways without re-running inference.
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
    """Dense depth result; raw map plus colorization via :meth:`plot`."""

    _COLORMAPS = {
        "inferno": "COLORMAP_INFERNO", "viridis": "COLORMAP_VIRIDIS",
        "magma": "COLORMAP_MAGMA", "plasma": "COLORMAP_PLASMA", "jet": "COLORMAP_JET",
    }

    def __init__(self, *, orig_img: np.ndarray, depth: np.ndarray):
        self.orig_img = orig_img
        self.depth = depth

    def __repr__(self) -> str:
        return f"DepthResult(shape={tuple(self.depth.shape)})"

    def normalized(self) -> np.ndarray:
        d = self.depth.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-8:
            return np.zeros_like(d)
        return (d - lo) / (hi - lo)

    def plot(self, *, colormap: str = "inferno") -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for DepthResult.plot().")
        norm = (self.normalized() * 255).astype(np.uint8)
        cmap = getattr(cv2, self._COLORMAPS.get(colormap, "COLORMAP_INFERNO"))
        return cv2.applyColorMap(norm, cmap)

    def to_dict(self) -> Dict[str, Any]:
        return {"task": "depth", "shape": list(self.depth.shape)}


# --------------------------------------------------------------------------- #
# TrackResult
# --------------------------------------------------------------------------- #
class TrackResult:
    """Tracker output: instances each carrying a persistent ``id``."""

    def __init__(self, *, instances: List[Instance],
                 orig_img: Optional[np.ndarray] = None,
                 rendered: Optional[np.ndarray] = None,
                 raw: Optional[list] = None):
        self.instances = instances
        self.orig_img = orig_img
        # ``rendered`` is the tracker's own rich overlay (student box, trails, etc.).
        self.rendered = rendered
        # ``raw`` is the backend's raw target rows: [x1,y1,x2,y2,id,(cls),(conf)].
        self.raw = raw if raw is not None else []

    def __iter__(self):
        return iter(self.instances)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]

    def __repr__(self) -> str:
        return f"TrackResult(tracks={len(self.instances)})"

    @property
    def ids(self) -> List[int]:
        return [i.id for i in self.instances if i.id is not None]

    @property
    def boxes(self) -> np.ndarray:
        boxes = [i.box for i in self.instances if i.box is not None]
        return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), np.float32)

    def plot(self, frame: np.ndarray = None, *, boxes: bool = True, labels: bool = True,
             color: tuple = (255, 0, 0), thickness: int = 2) -> np.ndarray:
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
        return {
            "task": "track",
            "tracks": [
                {"id": i.id,
                 "bbox": (np.asarray(i.box).tolist() if i.box is not None else None),
                 "confidence": i.confidence, "cls": i.cls}
                for i in self.instances
            ],
        }
