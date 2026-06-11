"""
Depth Estimation Module
Provides high-level wrapper for depth estimation models.
"""

from . import DepthAnythingV2Inference, Models
from ..results import DepthResult
import os
import numpy as np
from typing import Optional, Tuple, Union, List


class DepthBase:
    """Base class for depth estimation."""
    default_model = None

    def __init__(self, model=None, device='cpu', input_size: int = 518,
                 verbose: bool = True, **kwargs):
        """
        Initialize depth estimator.

        Args:
            model: Model enum from Models.Depth (e.g., Models.Depth.DepthAnythingV2.vitl)
            device: Device to run inference on (0, 'cuda', 'cpu', 'mps')
            input_size: Input image size for inference (default: 518)
            verbose: Whether to print initialization info
        """
        if model is None:
            if self.default_model is None:
                raise ValueError("Model must be provided either as parameter or class attribute")
            model = self.default_model

        Models.validate_depth_model(model)

        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)

        self.minfo = Models._get_model_info(model)
        self.depth_framework = self.minfo['backend']
        print(f'Initiating {self.depth_framework} {model.name} for Depth Estimation')

        # Get model configuration
        model_config = Models.get_depth_config(model)

        if self.depth_framework == 'DepthAnythingV2':
            self.depth_estimator = DepthAnythingV2Inference(
                model_path=model_path,
                model_config=model_config,
                device=device,
                input_size=input_size,
                verbose=verbose
            )
        else:
            raise ValueError(f"Invalid depth model type: {self.depth_framework}")

        self.model = model
        self.device = device
        self.input_size = input_size
        self.verbose = verbose

    def predict(self, source) -> Union[DepthResult, List[DepthResult]]:
        """Estimate depth for an image or a list of images.

        Args:
            source: a single BGR frame (HxWx3) or a list of frames.

        Returns:
            A :class:`~physiotrack.results.DepthResult` for a single frame, or a
            ``list[DepthResult]`` for a list. Get the raw map via ``result.depth``,
            a colorized view via ``result.plot(colormap=...)``, and ``0..1`` data
            via ``result.normalized()``.
        """
        if isinstance(source, (list, tuple)):
            depths = self.depth_estimator.inference_batch(list(source), False, None)
            return [DepthResult(orig_img=frame, depth=depth)
                    for frame, depth in zip(source, depths)]
        depth = self.depth_estimator.inference(source, False, None)
        return DepthResult(orig_img=source, depth=depth)

    def __call__(self, source):
        return self.predict(source)

    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if hasattr(self.depth_estimator, 'get_avg_inference_time'):
            return self.depth_estimator.get_avg_inference_time()
        return 0.0

    def get_avg_fps(self) -> float:
        """Get average FPS based on inference times."""
        if hasattr(self.depth_estimator, 'get_avg_fps'):
            return self.depth_estimator.get_avg_fps()
        return 0.0


class Depth:
    """
    High-level interface for depth estimation.

    Usage:
        depth = Depth.DepthAnythingV2Base(device=0)
        result = depth.predict(frame)            # or depth(frame)
        raw = result.depth                       # raw float depth map (HxW)
        colored = result.plot(colormap='inferno')
        norm = result.normalized()               # 0..1
    """

    class Custom(DepthBase):
        """Custom depth estimation with user-specified model."""

        def __init__(self, model, device='cpu', input_size: int = 518,
                     verbose: bool = True, **kwargs):
            """
            Initialize custom depth estimator.

            Args:
                model: Model enum from Models.Depth (e.g., Models.Depth.DepthAnythingV2.vitl)
                device: Device to run inference on (0, 'cuda', 'cpu', 'mps')
                input_size: Input image size for inference (default: 518)
                verbose: Whether to print initialization info
            """
            Models.validate_depth_model(model)
            super().__init__(
                model=model,
                device=device,
                input_size=input_size,
                verbose=verbose,
                **kwargs
            )

    class DepthAnythingV2(DepthBase):
        """DepthAnythingV2 depth estimation with default Large model."""
        default_model = Models.Depth.DepthAnythingV2.vitl

    class DepthAnythingV2Small(DepthBase):
        """DepthAnythingV2 depth estimation with Small model (faster, less accurate)."""
        default_model = Models.Depth.DepthAnythingV2.vits

    class DepthAnythingV2Base(DepthBase):
        """DepthAnythingV2 depth estimation with Base model (balanced)."""
        default_model = Models.Depth.DepthAnythingV2.vitb

    class DepthAnythingV2Large(DepthBase):
        """DepthAnythingV2 depth estimation with Large model (most accurate)."""
        default_model = Models.Depth.DepthAnythingV2.vitl
