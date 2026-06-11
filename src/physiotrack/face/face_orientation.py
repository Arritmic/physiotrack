"""
Face orientation estimation using 6DRepNet360.
"""
import os
import numpy as np
from typing import Optional, Union, List
from ..modules._6DRepNet360 import HeadPoseEstimator
from ..models import Models
from ..results import Result, Instance


class FaceOrientation:
    """
    Face orientation estimator using 6DRepNet360.
    
    This class provides head pose estimation (roll, pitch, yaw) from face images
    to determine the orientation of the face in 3D space using the 6D rotation representation network.
    
    Args:
        model: Face orientation model from ``Models.Pose3D.FaceOrientation``
               (default: ``Models.Pose3D.FaceOrientation.default``).
        device: 'cpu', 'cuda', or device id (e.g. ``0``). Default: 'cpu'.
        verbose: Whether to print verbose output (default: False).

    Example:
        >>> from physiotrack import FaceOrientation, VRFace
        >>> import cv2
        >>>
        >>> face = VRFace(device='cuda')
        >>> orient = FaceOrientation(device='cuda')
        >>>
        >>> frame = cv2.imread('image.jpg')
        >>> boxes = face.predict(frame).boxes        # (N, 4)
        >>> result = orient.predict(frame, boxes)    # Result(task="face")
        >>>
        >>> for inst in result:
        ...     print(inst.orientation)              # {"yaw":.., "pitch":.., "roll":..}
        >>> annotated = result.plot()                # draws pose axes
    """
    
    def __init__(self,
                 model=None,
                 device: Union[str, int] = 'cpu',
                 verbose: bool = False,
                 **kwargs):

        # Validate and get model
        if model is None:
            model = Models.Pose3D.FaceOrientation.default

        Models.validate_pose3d_model(model, expected_subclass='FaceOrientation')

        # Check if model file exists, download if needed
        model_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'modules',
            'model_data',
            model.value
        )
        if not os.path.isfile(model_path):
            # Try to download from HuggingFace, but skip if not available
            # The 6DRepNet360 model will auto-download from its source
            try:
                Models.download_model(model)
            except Exception as e:
                if verbose:
                    print(f"Note: Could not download from HuggingFace ({e}). Model will auto-download from 6DRepNet source.")

        # Rendering is handled by Result.plot(); the backend never draws.
        self.estimator = HeadPoseEstimator(
            model=model,
            device=device,
            render_pose=False,
            verbose=verbose,
            **kwargs
        )

        self.model = model
        self.device = device
        self.verbose = verbose

    def __call__(self, img: np.ndarray, bboxes: np.ndarray = None):
        return self.predict(img, bboxes)

    @staticmethod
    def _to_result(frame, results_dict) -> Result:
        instances = []
        for det in (results_dict or {}).get("detections", []):
            instances.append(Instance(
                id=det.get("id"),
                box=(np.array(det["bbox"], dtype=np.float32) if det.get("bbox") is not None else None),
                orientation=det.get("pose"),
            ))
        return Result(orig_img=frame, instances=instances, task="face")

    def predict(self, img: np.ndarray, bboxes: np.ndarray = None) -> Result:
        """Estimate head orientation (yaw/pitch/roll) for faces in a frame.

        Args:
            img: BGR image.
            bboxes: face boxes ``[N, 4]`` as ``[x1, y1, x2, y2]``; if ``None`` the
                whole image is treated as one face.

        Returns:
            A :class:`Result` (task="face"); each ``instance.orientation`` is a dict
            ``{"yaw", "pitch", "roll"}``. ``result.plot()`` draws the pose axes.
        """
        _, results_dict = self.estimator.predict(img, bboxes)
        return self._to_result(img, results_dict)

    def predict_batch(self, frames, bboxes_list=None) -> List[Result]:
        """Batch variant of :meth:`predict`; returns ``list[Result]``."""
        outputs = self.estimator.predict_batch(frames, bboxes_list)
        return [self._to_result(frame, results_dict)
                for frame, (_, results_dict) in zip(frames, outputs)]
    
    def get_avg_inference_time(self):
        """Get average inference time in milliseconds."""
        return self.estimator.get_avg_inference_time()
    
    def get_avg_fps(self):
        """Get average FPS based on inference times."""
        return self.estimator.get_avg_fps()

