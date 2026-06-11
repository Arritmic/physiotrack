from . import Detector, Models
from ..results import Result, Instance
import os
import numpy as np


class _DetectionAPI:
    """Mixin adding the unified ``predict()`` -> :class:`Result` interface.

    Mixed in ahead of the YOLO ``Detector`` backend so ``predict``/``__call__`` here
    take precedence over the backend's raw ``detect``/``__call__``.
    """

    _task = "detect"

    def predict(self, source, *, conf=None, iou=None, classes=None):
        """Run detection on an image or a list of images.

        Args:
            source: a single BGR frame (HxWx3) or a list of frames.
            conf: optional per-call confidence threshold override.
            iou: optional per-call NMS/IoU threshold override.
            classes: optional per-call class filter override.

        Returns:
            A :class:`Result` for a single frame, or ``list[Result]`` for a list.
        """
        overrides = {}
        if conf is not None:
            overrides["conf"] = conf
        if iou is not None:
            overrides["iou"] = iou
        if classes is not None:
            overrides["classes"] = classes

        if isinstance(source, (list, tuple)):
            outputs = self.detect_batch(list(source), **overrides)
            return [self._to_result(frame, results)
                    for frame, (results, _) in zip(source, outputs)]

        results, _ = self.detect(source, **overrides)
        return self._to_result(source, results)

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def _to_result(self, frame, results) -> Result:
        names = getattr(self.model, "names", None)
        instances = []
        r0 = results[0] if results else None
        if r0 is not None and getattr(r0, "boxes", None) is not None:
            data = r0.boxes.data.cpu().numpy()  # (N, 6): x1,y1,x2,y2,conf,cls
            for row in data:
                cls = int(row[5])
                cls_name = (names.get(cls) if isinstance(names, dict)
                            else (names[cls] if names is not None else None))
                instances.append(Instance(
                    box=np.array(row[:4], dtype=np.float32),
                    confidence=float(row[4]),
                    cls=cls,
                    cls_name=cls_name,
                ))
        return Result(orig_img=frame, instances=instances, task=self._task, names=names)


class ValidatedDetector(_DetectionAPI, Detector):
    expected_subclass = None
    model = None
    classes = None

    def __init__(self, model=None, *, conf=0.25, iou=0.45, classes=None,
                 device='cpu', verbose=False, **kwargs):
        if self.expected_subclass is None:
            raise NotImplementedError("expected_subclass must be set in subclass")

        if model is None:
            model = self.model
        if self.classes:
            classes = self.classes

        if model is None:
            raise ValueError("Model must be provided either as parameter or class attribute")

        Models.validate_det_model(model, expected_subclass=self.expected_subclass)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)
        super().__init__(
            model=model,
            device=device,
            OBJECTNESS_CONFIDENCE=conf,
            NMS_THRESHOLD=iou,
            classes=classes,
            render_labels=False,
            render_box_detections=False,
            verbose=verbose,
            **kwargs,
        )


class Detection:
    class Custom(_DetectionAPI, Detector):
        def __init__(self, model, *, conf=0.25, iou=0.45, classes=None,
                     device='cpu', verbose=False, **kwargs):
            Models.validate_det_model(model)
            model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
            if not os.path.isfile(model_path):
                Models.download_model(model)
            super().__init__(
                model=model,
                device=device,
                OBJECTNESS_CONFIDENCE=conf,
                NMS_THRESHOLD=iou,
                classes=classes,
                render_labels=False,
                render_box_detections=False,
                verbose=verbose,
                **kwargs,
            )

    class VRStudent(ValidatedDetector):
        expected_subclass = "VRStudent"
        model = Models.Detection.YOLO.VRSTUDENT.m_vrstudent

    class Face(ValidatedDetector):
        expected_subclass = "Face"
        model = Models.Detection.YOLO.FACE.m_face

    class Person(ValidatedDetector):
        expected_subclass = "Person"
        classes = [0]
        model = Models.Detection.YOLO.PERSON.m_person

    class VR(ValidatedDetector):
        expected_subclass = "VR"
        model = Models.Detection.YOLO.VR.m_vr
