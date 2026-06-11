from . import YoloPose, VitInference, SapiensPoseEstimation, Models, Detection
from ..results import Result, Instance, Keypoints
import os
import numpy as np


class PoseBase:
    detector = None
    default_model = None
    detector_class = None

    def __init__(self, model=None, *, conf=0.25, iou=0.45, classes=None, device='cpu',
                 detector_model=None, detector_conf=0.25, detector_iou=0.45,
                 verbose=False, **kwargs):
        if model is None:
            if self.default_model is None:
                raise ValueError("Model must be provided either as parameter or class attribute")
            model = self.default_model
        Models.validate_pose_model(model)

        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)

        self.minfo = Models._get_model_info(model)
        self.architecture = self.minfo['enum_class'].upper()
        self.pose_framework = self.minfo['backend']
        print(f'Initiating {self.pose_framework} {model.name} for the Pose estimation')

        # Pose backend. Rendering is delegated to Result.plot(), so overlay/draw flags
        # are off here.
        if self.pose_framework == 'ViTPose':
            self.pose_estimator = VitInference(model, device, False)
        elif self.pose_framework == 'YOLO':
            self.pose_estimator = YoloPose(model, device, conf, iou, classes,
                                           False, False, False, verbose, **kwargs)
        elif self.pose_framework == 'Sapiens':
            self.pose_estimator = SapiensPoseEstimation(model, device)
        else:
            raise ValueError("Invalid model type. Please check the configuration")

        # Optional person detector used when boxes aren't supplied to predict().
        if self.detector_class is not None:
            self.detector = self.detector_class(
                model=detector_model, conf=detector_conf, iou=detector_iou,
                classes=classes, device=device, verbose=verbose,
            )

        self.detector_model = detector_model
        self.device = device
        self.detector_conf = detector_conf
        self.detector_iou = detector_iou
        self.classes = classes
        self.verbose = verbose

    def _ensure_detector(self):
        if self.detector is None:
            if self.detector_class is not None:
                self.detector = self.detector_class()
            else:
                self.detector = Detection.Person(
                    device=self.device, conf=self.detector_conf,
                    iou=self.detector_iou, classes=self.classes, verbose=self.verbose,
                )
        return self.detector

    def _to_result(self, frame, frame_data) -> Result:
        instances = []
        for det in (frame_data or {}).get('detections', []):
            kps = (Keypoints(det['keypoints'], self.architecture)
                   if det.get('keypoints') else None)
            box = (np.array(det['bbox'], dtype=np.float32)
                   if det.get('bbox') is not None else None)
            instances.append(Instance(id=det.get('id'), box=box, keypoints=kps))
        return Result(orig_img=frame, instances=instances, task='pose',
                      architecture=self.architecture)

    def predict(self, source, boxes=None):
        """Estimate 2D pose on an image or a list of images.

        Args:
            source: a single BGR frame (HxWx3) or a list of frames.
            boxes: optional person boxes ``[[x1,y1,x2,y2], ...]``. If omitted (and the
                backend needs boxes), people are auto-detected with a person detector.

        Returns:
            A :class:`Result` (task="pose"); access keypoints via
            ``result[i].keypoints.by_name("left_wrist")``.
        """
        if isinstance(source, (list, tuple)):
            return self._predict_batch(list(source), boxes)
        return self._predict_one(source, boxes)

    def _predict_one(self, frame, boxes=None) -> Result:
        if boxes is None and self.pose_framework in ("ViTPose", "Sapiens"):
            boxes = self._ensure_detector().predict(frame).boxes.astype(int)
        _, frame_data = self.pose_estimator.inference(frame, boxes)
        return self._to_result(frame, frame_data)

    def _predict_batch(self, frames, boxes_list=None):
        if not isinstance(boxes_list, list) or len(boxes_list) != len(frames):
            boxes_list = [None] * len(frames)
        if hasattr(self.pose_estimator, 'inference_batch_frames'):
            # Auto-detect boxes where missing.
            resolved = []
            for frame, b in zip(frames, boxes_list):
                if b is None and self.pose_framework in ("ViTPose", "Sapiens"):
                    b = self._ensure_detector().predict(frame).boxes.astype(int)
                resolved.append(b)
            outputs = self.pose_estimator.inference_batch_frames(frames, resolved)
            return [self._to_result(frame, frame_data)
                    for frame, (_, frame_data) in zip(frames, outputs)]
        return [self._predict_one(f, b) for f, b in zip(frames, boxes_list)]

    def __call__(self, source, boxes=None):
        return self.predict(source, boxes)


class Pose:
    class Custom(PoseBase):
        def __init__(self, model, *, conf=0.25, iou=0.45, classes=None, device='cpu',
                     detector_model=None, detector_conf=0.25, detector_iou=0.45,
                     verbose=False, **kwargs):
            Models.validate_pose_model(model)
            super().__init__(model=model, conf=conf, iou=iou, classes=classes,
                             device=device, detector_model=detector_model,
                             detector_conf=detector_conf, detector_iou=detector_iou,
                             verbose=verbose, **kwargs)

    class VRStudent(PoseBase):
        detector_class = Detection.VRStudent
        default_model = Models.Pose.ViTPose.WholeBody.b_wholebody

    class Person(PoseBase):
        detector_class = Detection.Person
        default_model = Models.Pose.ViTPose.WholeBody.b_wholebody
