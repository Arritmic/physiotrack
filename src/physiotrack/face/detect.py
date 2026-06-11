"""
Face detection using YOLO detectors.
"""
from ..detect import ValidatedDetector
from ..models import Models


class Face(ValidatedDetector):
    """Face detector using YOLO models.

    Args:
        model: model from ``Models.Detection.YOLO.FACE`` (default: ``m_face``).
        conf: confidence threshold (default 0.25).
        iou: NMS/IoU threshold (default 0.45).
        device: 'cpu', 'cuda', or device id.

    Example:
        >>> from physiotrack import Face
        >>> face = Face(device='cuda')
        >>> result = face.predict(frame)          # or face(frame)
        >>> annotated = result.plot()
        >>> boxes = result.boxes
    """
    expected_subclass = "Face"
    model = Models.Detection.YOLO.FACE.m_face
    _task = "face"


class VRFace(ValidatedDetector):
    """Face detector tuned for VR-headset scenarios (YOLOv12l-face).

    Args:
        model: model from ``Models.Detection.YOLO.VRFACE`` (default: ``l_vrface``).
        conf: confidence threshold (default 0.25).
        iou: NMS/IoU threshold (default 0.45).
        device: 'cpu', 'cuda', or device id.

    Example:
        >>> from physiotrack import VRFace
        >>> face = VRFace(device='cuda')
        >>> result = face.predict(frame)
    """
    expected_subclass = "VRFace"
    model = Models.Detection.YOLO.VRFACE.l_vrface
    _task = "face"
