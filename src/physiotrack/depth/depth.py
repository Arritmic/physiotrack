"""
Depth Estimation Module
Provides high-level wrapper for depth estimation models.
"""

from . import DepthAnythingV2Inference, Models
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

    def estimate(self, frame: np.ndarray, normalize: bool = False,
                 colormap: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate depth from a single image.

        Args:
            frame: Input BGR image (HxWx3 numpy array)
            normalize: If True, normalize depth to 0-255 range
            colormap: If provided, return colored depth map using this colormap
                     (e.g., 'inferno', 'viridis', 'magma', 'plasma', 'jet')

        Returns:
            If colormap is None: Raw depth map (HxW numpy array, float32)
            If colormap is provided: Tuple of (raw_depth, colored_depth)
        """
        return self.depth_estimator.inference(frame, normalize, colormap)

    def estimate_batch(self, frames: List[np.ndarray], normalize: bool = False,
                       colormap: Optional[str] = None) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Estimate depth from a batch of images.

        Args:
            frames: List of input BGR images
            normalize: If True, normalize depth to 0-255 range
            colormap: If provided, return colored depth maps

        Returns:
            List of depth results (same format as estimate())
        """
        return self.depth_estimator.inference_batch(frames, normalize, colormap)

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
        # Using a specific model
        depth_estimator = Depth.Custom(model=Models.Depth.DepthAnythingV2.vitl, device=0)
        depth_map = depth_estimator.estimate(frame)

        # Using default model (DepthAnythingV2 Large)
        depth_estimator = Depth.DepthAnythingV2(device=0)
        depth_map, colored_depth = depth_estimator.estimate(frame, colormap='inferno')

        # Get normalized depth
        depth_map = depth_estimator.estimate(frame, normalize=True)
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
