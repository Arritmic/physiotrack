"""Face segmentation / skin extraction via SegFace face-part parsing.

A small reusable, plotter-free component that runs SegFace (which detects faces
itself, CelebAMask-HQ 19 classes) and exposes the result several ways:

* a boolean **skin ROI mask** (image-resolution) -- e.g. to feed
  :class:`~physiotrack.signals.ppg.estimator.HeartRateEstimator` (the default rPPG
  ROI), or for any downstream use;
* a **skin canvas** (image-resolution, rest = background) -- the extracted skin
  only, e.g. to display live next to the HR signal; and
* a **parsing canvas** -- the *full* face parsing, every SegFace class colorized
  with the CelebAMask-HQ palette on a background canvas.

It runs the segmenter once per :meth:`analyze` call and derives all three outputs
from the same inference. By default the ``skin`` class is used for the ROI; pass
``skin_classes`` to include others (e.g. ``("skin", "neck")``).

Example::

    fs = FaceSkinExtractor(device=0)
    skin_mask, skin_canvas, parsing = fs.analyze(frame)   # one SegFace pass
    est.update(frame, roi_mask=skin_mask)                 # rPPG on the segmented skin
"""

import cv2
import numpy as np
from typing import NamedTuple, Optional, Sequence, Tuple


class FaceParsing(NamedTuple):
    """Outputs of one SegFace pass over a frame."""
    skin_mask: np.ndarray       # bool (H, W) -- skin ROI (the ``skin_classes``)
    skin_canvas: np.ndarray     # BGR (H, W, 3) -- skin pixels only, rest = background
    parsing_canvas: np.ndarray  # BGR (H, W, 3) -- all face classes, palette-colorized
    seg_map: np.ndarray         # int (H, W) -- raw class-index map (all 19 classes)


class FaceSkinExtractor:
    """Segment the face with SegFace; expose skin ROI, skin canvas, full parsing."""

    def __init__(self,
                 segmenter=None,
                 *,
                 device: str = "cpu",
                 skin_classes: Sequence[str] = ("skin",),
                 background: Tuple[int, int, int] = (0, 0, 0),
                 verbose: bool = False):
        if segmenter is None:
            from physiotrack.segment import Segmentation
            segmenter = Segmentation.Face(device=device, verbose=verbose)
        self.segmenter = segmenter
        self.skin_classes = tuple(c.lower() for c in skin_classes)
        self.background = tuple(int(c) for c in background)
        self._class_idxs: Optional[list] = None

    # -- internals ---------------------------------------------------------
    def _predict(self, frame_bgr, boxes=None):
        return (self.segmenter.predict(frame_bgr, boxes=boxes) if boxes is not None
                else self.segmenter.predict(frame_bgr))

    def _resolve_idxs(self, names) -> list:
        if self._class_idxs is not None:
            return self._class_idxs
        items = names.items() if isinstance(names, dict) else enumerate(names or [])
        self._class_idxs = [int(i) for i, n in items if str(n).lower() in self.skin_classes]
        return self._class_idxs

    def _skin_mask_from(self, result, shape) -> np.ndarray:
        idxs = self._resolve_idxs(getattr(result, "names", None))
        seg = result.seg_map
        if not idxs or seg is None:
            return np.zeros(shape[:2], dtype=bool)
        return np.isin(seg, idxs)

    def _canvas_from_mask(self, frame_bgr, mask) -> np.ndarray:
        out = np.empty_like(frame_bgr)
        out[:] = self.background
        out[mask] = frame_bgr[mask]
        return out

    def _parsing_canvas_from(self, result, frame_bgr) -> np.ndarray:
        """Colorize every face class (seg > 0) onto a background canvas."""
        out = np.empty_like(frame_bgr)
        out[:] = self.background
        seg = result.seg_map
        if seg is None:
            return out
        fg = seg > 0
        palette = getattr(result, "palette", None)
        if palette is not None:
            palette = np.asarray(palette, dtype=np.uint8)
            idx = np.clip(seg, 0, len(palette) - 1)
            color = cv2.cvtColor(palette[idx], cv2.COLOR_RGB2BGR)   # palette is RGB
            out[fg] = color[fg]
        else:                                                       # no palette: show pixels
            out[fg] = frame_bgr[fg]
        return out

    # -- public API --------------------------------------------------------
    def skin_mask(self, frame_bgr, boxes=None) -> np.ndarray:
        """Boolean image-resolution mask of the segmented facial skin (ROI)."""
        result = self._predict(frame_bgr, boxes=boxes)
        return self._skin_mask_from(result, frame_bgr.shape)

    def canvas(self, frame_bgr, mask=None, boxes=None) -> np.ndarray:
        """Image-resolution canvas holding only the extracted skin (rest = background)."""
        if mask is None:
            mask = self.skin_mask(frame_bgr, boxes=boxes)
        return self._canvas_from_mask(frame_bgr, mask)

    def parsing_canvas(self, frame_bgr, boxes=None) -> np.ndarray:
        """Image-resolution canvas with the full face parsing (all classes colorized)."""
        result = self._predict(frame_bgr, boxes=boxes)
        return self._parsing_canvas_from(result, frame_bgr)

    def extract(self, frame_bgr, boxes=None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(skin_mask, skin_canvas)`` -- the ROI mask and the skin canvas."""
        result = self._predict(frame_bgr, boxes=boxes)
        mask = self._skin_mask_from(result, frame_bgr.shape)
        return mask, self._canvas_from_mask(frame_bgr, mask)

    def analyze(self, frame_bgr, boxes=None) -> FaceParsing:
        """Run SegFace once; return skin mask, skin canvas, full parsing canvas, seg map."""
        result = self._predict(frame_bgr, boxes=boxes)
        mask = self._skin_mask_from(result, frame_bgr.shape)
        seg = result.seg_map
        if seg is None:
            seg = np.zeros(frame_bgr.shape[:2], dtype=np.int32)
        return FaceParsing(
            skin_mask=mask,
            skin_canvas=self._canvas_from_mask(frame_bgr, mask),
            parsing_canvas=self._parsing_canvas_from(result, frame_bgr),
            seg_map=seg,
        )
