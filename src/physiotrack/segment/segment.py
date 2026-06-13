from . import Segmentor, SapiensSegmentation, draw_segmentation_map, Models
from ..results import Result, Instance
import os
import numpy as np


class SegmentationBase:
    default_model = None

    def __init__(self, model=None, *, conf=0.25, iou=0.45, classes=None,
                 device='cpu', filter=None, verbose=False, **kwargs):
        if model is None:
            if self.default_model is None:
                raise ValueError("Model must be provided either as parameter or class attribute")
            model = self.default_model

        Models.validate_seg_model(model)

        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)

        self.minfo = Models._get_model_info(model)
        self.segmentation_framework = self.minfo['backend']
        print(f'Initiating {self.segmentation_framework} {model.name} for Segmentation')

        if self.segmentation_framework == 'Yolo':
            self.segmentor = Segmentor(model, device, conf, iou, classes,
                                       False, verbose, **kwargs)
        elif self.segmentation_framework == 'Sapiens':
            self.segmentor = SapiensSegmentation(model, device)
        elif self.segmentation_framework == 'SegFace':
            from ..modules import SegFaceInference
            self.segmentor = SegFaceInference(model_path, input_resolution=512, device=device)
        else:
            raise ValueError("Invalid model type. Please check the configuration")

        self.model = model
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.verbose = verbose

        # Optional bbox-based filtering of the segmentation map.
        filter = filter or {}
        self.bbox_filter = filter.get('bbox_filter', False)
        self.detector_index = filter.get('detector_index', None)
        self.detector_class_filter = filter.get('detector_class_filter', None)

    # -- inference ----------------------------------------------------------- #
    def _segment_map(self, frame) -> np.ndarray:
        if self.segmentation_framework == 'Yolo':
            _, segmentation_map = self.segmentor.segment(frame)
        else:  # Sapiens
            segmentation_map = self.segmentor.inference(frame)
        return segmentation_map

    @staticmethod
    def _filter_by_boxes(segmentation_map, bboxes) -> np.ndarray:
        h, w = segmentation_map.shape[:2]
        filtered = np.zeros((h, w), dtype=segmentation_map.dtype)
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            filtered[y1:y2, x1:x2] = segmentation_map[y1:y2, x1:x2]
        return filtered

    def predict(self, source, boxes=None):
        """Run segmentation on an image or a list of images.

        Args:
            source: a single BGR frame (HxWx3) or a list of frames.
            boxes: optional ``[[x1,y1,x2,y2], ...]``; only segmentation inside these
                boxes is kept.

        Returns:
            A :class:`Result` (task="segment") whose ``.seg_map`` is the class-index
            map. ``result.plot()`` overlays the colorized segmentation.
        """
        if isinstance(source, (list, tuple)):
            return [self._predict_one(frame, boxes) for frame in source]
        return self._predict_one(source, boxes)

    def _predict_one(self, frame, boxes=None) -> Result:
        segmentation_map = self._segment_map(frame)
        if boxes is not None and len(boxes) > 0:
            segmentation_map = self._filter_by_boxes(segmentation_map, boxes)
        return Result(orig_img=frame, instances=[], task="segment",
                      seg_map=segmentation_map)

    def __call__(self, source, boxes=None):
        return self.predict(source, boxes)

    def get_avg_inference_time(self):
        """Get average inference time in milliseconds."""
        if hasattr(self.segmentor, 'get_avg_inference_time'):
            return self.segmentor.get_avg_inference_time()
        return 0.0

    def get_avg_fps(self):
        """Get average FPS based on inference times."""
        if hasattr(self.segmentor, 'get_avg_fps'):
            return self.segmentor.get_avg_fps()
        return 0.0


class Segmentation:
    class Custom(SegmentationBase):
        def __init__(self, model, *, conf=0.25, iou=0.45, classes=None,
                     device='cpu', filter=None, verbose=False, **kwargs):
            Models.validate_seg_model(model)
            super().__init__(model=model, conf=conf, iou=iou, classes=classes,
                             device=device, filter=filter, verbose=verbose, **kwargs)

    class VRHead(SegmentationBase):
        default_model = Models.Segmentation.YOLO.VRHEAD.M11

    class Person(SegmentationBase):
        default_model = Models.Segmentation.YOLO.PERSON.m_person

    class BodyPart(SegmentationBase):
        default_model = Models.Segmentation.Sapiens.BodyPart.B1_TS_SEG

    class Face(SegmentationBase):
        """SegFace face-part parsing (CelebAMask-HQ, 19 classes).

        Unlike the whole-frame segmenters, SegFace runs on face crops. If
        ``predict`` is called without ``boxes``, faces are auto-detected with a
        YOLO face detector (mirroring how ``Pose`` auto-detects people). The
        returned :class:`Result` carries a full-frame ``seg_map`` of face-part
        class indices; ``result.plot()`` overlays the parsing with the 19-class
        palette.
        """
        default_model = Models.Segmentation.SegFace.Face.swinb_celeba_512

        def __init__(self, model=None, *, device='cpu', face_detector=None,
                     face_conf=0.25, face_iou=0.45, verbose=False, **kwargs):
            super().__init__(model=model, device=device, verbose=verbose, **kwargs)
            self._face_detector = face_detector
            self._face_conf = face_conf
            self._face_iou = face_iou

        def _ensure_detector(self):
            if self._face_detector is None:
                from ..detect import Detection
                self._face_detector = Detection.Face(
                    conf=self._face_conf, iou=self._face_iou, device=self.device)
            return self._face_detector

        def _predict_one(self, frame, boxes=None) -> Result:
            from ..modules.SegFace import CELEBA_CLASSES, CELEBA_PALETTE
            if boxes is None:
                boxes = self._ensure_detector().predict(frame).boxes

            h, w = frame.shape[:2]
            seg_map = np.zeros((h, w), dtype=np.int32)
            instances = []
            for box in (boxes if boxes is not None else []):
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                parsing = self.segmentor.infer(frame[y1:y2, x1:x2])
                fg = parsing > 0
                seg_map[y1:y2, x1:x2][fg] = parsing[fg]
                instances.append(Instance(box=np.array([x1, y1, x2, y2], dtype=float)))

            names = {i: n for i, n in enumerate(CELEBA_CLASSES)}
            return Result(orig_img=frame, instances=instances, task="segment",
                          seg_map=seg_map, names=names, palette=CELEBA_PALETTE)
