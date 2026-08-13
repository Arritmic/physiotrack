"""
Face detection using YOLO detectors.
"""
from ..detect import ValidatedDetector
from ..models import Models


class Face(ValidatedDetector):
    """YOLO-face detector preset returning face bounding boxes.

    Wraps the default YOLO face model (``Models.Detection.YOLO.FACE.m_face``).
    Calling [`predict`][physiotrack.Face.predict] (or the instance directly)
    returns a [`Result`][physiotrack.Result] with ``task="face"`` whose instances
    carry face boxes. Pair it with
    [`FaceOrientation`][physiotrack.FaceOrientation] to add head pose.

    This is the face-specific entry point and therefore returns ``task="face"``.
    [`Detection.Face`][physiotrack.Detection.Face] uses the same default weights
    but belongs to the generic detection namespace and returns ``task="detect"``.

    Args:
        model (Models.Detection.YOLO.FACE, optional): Face model variant.
            Defaults to ``None`` (uses ``Models.Detection.YOLO.FACE.m_face``).
        conf (float, optional): Confidence threshold in ``[0.0, 1.0]``. Defaults
            to ``0.25``.
        iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]``. Defaults to
            ``0.45``.
        classes (list[int], optional): Class-id filter. Defaults to ``None``.
        device (str | int, optional): ``"cpu"``, ``"cuda"``, or a device id (e.g.
            ``0``). Defaults to ``"cpu"``.
        verbose (bool, optional): Print backend progress. Defaults to ``False``.

    Example:
        ```python
        import physiotrack as pt
        face = pt.Face(device="cuda")
        result = face.predict(frame)          # or face(frame)
        annotated = result.plot()
        boxes = result.boxes                  # (N, 4)
        ```

    Note:
        The first call for a validated model auto-downloads its weights from
        Hugging Face.

    See Also:
        [`Detection.Face`][physiotrack.Detection.Face]: the generic-detector entry
            point using the same default checkpoint.
        [`VRFace`][physiotrack.VRFace]: VR-headset-tuned face detector.
        [`FaceOrientation`][physiotrack.FaceOrientation]: head pose from face boxes.
    """
    expected_subclass = "Face"
    model = Models.Detection.YOLO.FACE.m_face
    _task = "face"


class VRFace(ValidatedDetector):
    """Face detector preset tuned for VR-headset scenarios (YOLOv12l-face).

    Same interface as [`Face`][physiotrack.Face], but wraps the VR-tuned model
    (``Models.Detection.YOLO.VRFACE.l_vrface``) which is more robust to partial
    occlusion from head-mounted displays. Returns a
    [`Result`][physiotrack.Result] with ``task="face"``.

    Args:
        model (Models.Detection.YOLO.VRFACE, optional): Face model variant.
            Defaults to ``None`` (uses ``Models.Detection.YOLO.VRFACE.l_vrface``).
        conf (float, optional): Confidence threshold in ``[0.0, 1.0]``. Defaults
            to ``0.25``.
        iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]``. Defaults to
            ``0.45``.
        classes (list[int], optional): Class-id filter. Defaults to ``None``.
        device (str | int, optional): ``"cpu"``, ``"cuda"``, or a device id.
            Defaults to ``"cpu"``.
        verbose (bool, optional): Print backend progress. Defaults to ``False``.

    Example:
        ```python
        import physiotrack as pt
        face = pt.VRFace(device="cuda")
        boxes = face.predict(frame).boxes     # (N, 4)
        ```

    Note:
        The first call for a validated model auto-downloads its weights from
        Hugging Face.

    See Also:
        [`Face`][physiotrack.Face]: general-purpose YOLO face detector.
    """
    expected_subclass = "VRFace"
    model = Models.Detection.YOLO.VRFACE.l_vrface
    _task = "face"
