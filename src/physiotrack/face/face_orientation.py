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
    """Head pose (yaw/pitch/roll) estimator using 6DRepNet360.

    Estimates the 3D orientation of each face using the 6D rotation
    representation network. Given a frame and a set of face boxes (typically from
    [`Face`][physiotrack.Face] or [`VRFace`][physiotrack.VRFace]), it returns a
    [`Result`][physiotrack.Result] with ``task="face"`` whose instances carry an
    ``orientation`` dict ``{"yaw", "pitch", "roll"}`` in degrees;
    ``result.plot()`` draws the corresponding pose axes.

    Attributes:
        estimator (HeadPoseEstimator): The underlying 6DRepNet360 backend
            (rendering disabled — overlays are drawn by ``Result.plot()``).
        model: The resolved ``Models.Pose3D.FaceOrientation`` variant in use.
        device (str | int): Compute device (``"cpu"``, ``"cuda"``, or device id).
        verbose (bool): Whether backend progress is printed.

    Example:
        ```python
        import physiotrack as pt
        import cv2

        face = pt.VRFace(device="cuda")
        orient = pt.FaceOrientation(device="cuda")

        frame = cv2.imread("image.jpg")
        boxes = face.predict(frame).boxes         # (N, 4)
        result = orient.predict(frame, boxes)     # Result(task="face")
        for inst in result:
            print(inst.orientation)               # {"yaw":.., "pitch":.., "roll":..}
        annotated = result.plot()                 # draws pose axes
        ```

    See Also:
        [`VRFace`][physiotrack.VRFace]: face detector to source the boxes.
        [`Result`][physiotrack.Result]: the returned per-frame object.
    """

    def __init__(self,
                 model=None,
                 device: Union[str, int] = 'cpu',
                 verbose: bool = False,
                 **kwargs):
        """Construct a head-pose estimator.

        Args:
            model (Models.Pose3D.FaceOrientation, optional): Orientation model
                variant. Defaults to ``None`` (uses
                ``Models.Pose3D.FaceOrientation.default``).
            device (str | int, optional): ``"cpu"``, ``"cuda"``, or a device id
                (e.g. ``0``). Defaults to ``"cpu"``.
            verbose (bool, optional): Print progress/download notes. Defaults to
                ``False``.
            **kwargs (Any): Extra keyword arguments forwarded to the underlying
                ``HeadPoseEstimator`` backend.

        Note:
            If the model weights are missing they are auto-downloaded (from
            Hugging Face, or the 6DRepNet source as a fallback) on construction.
        """
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
        """Alias for [`predict`][physiotrack.FaceOrientation.predict].

        Args:
            img (np.ndarray): BGR frame ``(H, W, 3)``.
            bboxes (np.ndarray, optional): Face boxes ``(N, 4)`` as
                ``[x1, y1, x2, y2]``. Defaults to ``None`` (whole image is one
                face).

        Returns:
            Result: Same as [`predict`][physiotrack.FaceOrientation.predict].
        """
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
            img (np.ndarray): BGR frame of shape ``(H, W, 3)``.
            bboxes (np.ndarray, optional): Face boxes of shape ``(N, 4)`` as
                ``[x1, y1, x2, y2]``. Defaults to ``None`` (the whole image is
                treated as a single face).

        Returns:
            Result: A [`Result`][physiotrack.Result] with ``task="face"``; each
                ``instance.orientation`` is a dict ``{"yaw", "pitch", "roll"}`` in
                degrees. ``result.plot()`` draws the pose axes.

        Example:
            ```python
            import physiotrack as pt
            orient = pt.FaceOrientation()
            boxes = pt.Face().predict(frame).boxes
            for inst in orient.predict(frame, boxes):
                print(inst.orientation)
            ```
        """
        _, results_dict = self.estimator.predict(img, bboxes)
        return self._to_result(img, results_dict)

    def predict_batch(self, frames, bboxes_list=None) -> List[Result]:
        """Estimate head orientation for a batch of frames.

        Args:
            frames (list[np.ndarray]): List of BGR frames, each ``(H, W, 3)``.
            bboxes_list (list[np.ndarray], optional): Per-frame face boxes, each
                ``(N, 4)``. Defaults to ``None`` (each whole frame is one face).

        Returns:
            list[Result]: One [`Result`][physiotrack.Result] per input frame, in
                order; see [`predict`][physiotrack.FaceOrientation.predict].
        """
        outputs = self.estimator.predict_batch(frames, bboxes_list)
        return [self._to_result(frame, results_dict)
                for frame, (_, results_dict) in zip(frames, outputs)]

    def get_avg_inference_time(self):
        """Return the average per-call inference time.

        Returns:
            float: Mean inference time in milliseconds across processed calls.
        """
        return self.estimator.get_avg_inference_time()

    def get_avg_fps(self):
        """Return the average throughput derived from inference times.

        Returns:
            float: Mean frames-per-second across processed calls.
        """
        return self.estimator.get_avg_fps()

