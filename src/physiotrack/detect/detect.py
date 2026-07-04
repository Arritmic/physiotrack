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
        """Run object detection on one image or a batch of images.

        Accepts a single BGR frame or a list of frames. Any of ``conf``, ``iou``
        or ``classes`` supplied here override, for this call only, the values set
        on the model at construction time.

        Args:
            source (np.ndarray | list[np.ndarray] | tuple[np.ndarray]): A single
                BGR image ``(H, W, 3)`` or a list/tuple of such frames for batch
                inference.
            conf (float, optional): Per-call confidence threshold in ``[0.0, 1.0]``.
                Defaults to ``None`` (uses the model's construction-time ``conf``,
                normally ``0.25``).
            iou (float, optional): Per-call NMS/IoU threshold in ``[0.0, 1.0]``.
                Defaults to ``None`` (uses the model's construction-time ``iou``,
                normally ``0.45``).
            classes (list[int], optional): Restrict detections to these class ids.
                Defaults to ``None`` (uses the model's construction-time class
                filter, if any).

        Returns:
            Result | list[Result]: A [`Result`][physiotrack.Result] whose
                ``.boxes`` / ``.instances`` hold the detections for a single
                frame, or a ``list[Result]`` (one per frame) when ``source`` is a
                list or tuple.

        Example:
            ```python
            import physiotrack as pt

            det = pt.Detection.Person(conf=0.25, device=0)
            result = det.predict(frame)          # or: det(frame)
            boxes = result.boxes                 # (N, 4) xyxy
            annotated = result.plot()
            ```

        See Also:
            [`Result`][physiotrack.Result]: the returned detection container.
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
        """Alias for [`predict`][physiotrack.Detection].

        Lets a detector instance be called directly, e.g. ``det(frame)`` instead
        of ``det.predict(frame)``. All keyword arguments are forwarded to
        ``predict`` (``conf``, ``iou``, ``classes``).

        Args:
            source (np.ndarray | list[np.ndarray]): A single BGR frame ``(H, W, 3)``
                or a list of frames.
            **kwargs (Any): Forwarded to [`predict`][physiotrack.Detection].

        Returns:
            Result | list[Result]: See [`predict`][physiotrack.Detection].
        """
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
    """Base for the built-in detection presets with a validated default model.

    Subclasses (``Detection.Person``, ``Detection.Face``, ``Detection.VR``,
    ``Detection.VRStudent``) set ``model`` to a specific ``Models.Detection.*``
    enum and ``expected_subclass`` to the enum group the model must belong to,
    so construction fails fast if an incompatible model is passed. The chosen
    weights are auto-downloaded on first use if not already cached.

    Not instantiated directly; use one of the ``Detection.*`` presets.
    """

    expected_subclass = None
    model = None
    classes = None

    def __init__(self, model=None, *, conf=0.25, iou=0.45, classes=None,
                 device='cpu', verbose=False, **kwargs):
        """Configure a preset detector.

        Args:
            model (Models.Detection.*, optional): Override the preset's default
                weights with another validated detection model enum. Defaults to
                ``None`` (uses the preset's class-level ``model``).
            conf (float, optional): Objectness confidence threshold in
                ``[0.0, 1.0]``. Defaults to ``0.25``.
            iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]``. Defaults
                to ``0.45``.
            classes (list[int], optional): Restrict detections to these class ids.
                Defaults to ``None`` (or the preset's class-level ``classes``,
                e.g. ``[0]`` for ``Detection.Person``).
            device (int | str, optional): Inference device, e.g. ``'cpu'``,
                ``'cuda'``, ``'mps'`` or a device index like ``0``. Defaults to
                ``'cpu'``.
            verbose (bool, optional): Print backend inference logs. Defaults to
                ``False``.
            **kwargs (Any): Additional keyword arguments forwarded to the underlying
                YOLO ``Detector`` backend.

        Raises:
            NotImplementedError: If used on a subclass that does not set
                ``expected_subclass``.
            ValueError: If no model is provided or resolvable from the class.

        Note:
            On the first use of a given model the weights are auto-downloaded
            from Hugging Face (except stock Ultralytics weights) and cached.
        """
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
    """Object-detection predictors, grouped as ready-to-use presets.

    ``Detection`` is a namespace of nested predictor classes. Instantiate a
    preset (or [`Detection.Custom`][physiotrack.Detection.Custom]) to build a
    model, then call [`predict`][physiotrack.Detection] (or the instance
    directly) on a frame. Each predictor returns a
    [`Result`][physiotrack.Result].

    Presets:
        - [`Person`][physiotrack.Detection.Person]: person detection (COCO class 0).
        - [`Face`][physiotrack.Detection.Face]: face detection.
        - [`VR`][physiotrack.Detection.VR]: VR-headset object detection.
        - [`VRStudent`][physiotrack.Detection.VRStudent]: VR-student detection.
        - [`Custom`][physiotrack.Detection.Custom]: any validated detection model.

    Example:
        ```python
        import physiotrack as pt

        det = pt.Detection.Person(conf=0.25, device=0)   # also .Face() .VR() .VRStudent()
        result = det.predict(frame)                      # -> pt.Result
        annotated = result.plot()
        ```

    Note:
        Weights are auto-downloaded from Hugging Face on first use and cached.

    See Also:
        [`Result`][physiotrack.Result]: detection output container.
        [`Segmentation`][physiotrack.Segmentation]: mask-based prediction.
    """

    class Custom(_DetectionAPI, Detector):
        """Detector backed by any user-specified validated detection model.

        Use this when you want to run a specific ``Models.Detection.*`` variant
        rather than one of the fixed presets.

        Example:
            ```python
            import physiotrack as pt
            from physiotrack import Models

            det = pt.Detection.Custom(model=Models.Detection.YOLO.VR.m_vr)
            result = det.predict(frame)
            ```
        """

        def __init__(self, model, *, conf=0.25, iou=0.45, classes=None,
                     device='cpu', verbose=False, **kwargs):
            """Configure a custom detector.

            Args:
                model (Models.Detection.*): A validated detection model enum,
                    e.g. ``Models.Detection.YOLO.VR.m_vr``.
                conf (float, optional): Objectness confidence threshold in
                    ``[0.0, 1.0]``. Defaults to ``0.25``.
                iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]``.
                    Defaults to ``0.45``.
                classes (list[int], optional): Restrict detections to these class
                    ids. Defaults to ``None`` (all classes).
                device (int | str, optional): Inference device, e.g. ``'cpu'``,
                    ``'cuda'``, ``'mps'`` or a device index. Defaults to ``'cpu'``.
                verbose (bool, optional): Print backend inference logs. Defaults
                    to ``False``.
                **kwargs (Any): Additional keyword arguments forwarded to the
                    underlying YOLO ``Detector`` backend.

            Note:
                On first use the model weights are auto-downloaded and cached.
            """
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
        """VR-student detector.

        Wraps ``Models.Detection.YOLO.VRSTUDENT.m_vrstudent``. See
        [`ValidatedDetector`][physiotrack.Detection] for
        constructor arguments.
        """
        expected_subclass = "VRStudent"
        model = Models.Detection.YOLO.VRSTUDENT.m_vrstudent

    class Face(ValidatedDetector):
        """Face detector.

        Wraps ``Models.Detection.YOLO.FACE.m_face``. See
        [`ValidatedDetector`][physiotrack.Detection] for
        constructor arguments.
        """
        expected_subclass = "Face"
        model = Models.Detection.YOLO.FACE.m_face

    class Person(ValidatedDetector):
        """Person detector (COCO class ``0`` only).

        Wraps ``Models.Detection.YOLO.PERSON.m_person`` and pins the class filter
        to ``[0]`` (person). See
        [`ValidatedDetector`][physiotrack.Detection] for
        constructor arguments.
        """
        expected_subclass = "Person"
        classes = [0]
        model = Models.Detection.YOLO.PERSON.m_person

    class VR(ValidatedDetector):
        """VR-headset object detector.

        Wraps ``Models.Detection.YOLO.VR.m_vr``. See
        [`ValidatedDetector`][physiotrack.Detection] for
        constructor arguments.
        """
        expected_subclass = "VR"
        model = Models.Detection.YOLO.VR.m_vr
