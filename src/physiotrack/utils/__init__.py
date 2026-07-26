"""Shared utilities: logging, display helpers, spatial transforms, and
segmentation-map post-processing."""

from .display_utils import get_screen_size, resize_frame_for_display
from .logger import log_to_file
from .sapiens_post_process import (
    combine_contours,
    draw_class_contours,
    exclude_contours,
    filter_by_box,
    filter_by_connectivity,
    filter_contours_by_area,
    process_segmentation_map,
    remove_isolated_contours,
)
from .spatial_transforms import compute_homography, get_foot_position, transform_point

__all__ = [
    # logging / display
    "log_to_file",
    "get_screen_size",
    "resize_frame_for_display",
    # spatial transforms
    "compute_homography",
    "transform_point",
    "get_foot_position",
    # segmentation-map post-processing
    "filter_by_box",
    "exclude_contours",
    "combine_contours",
    "remove_isolated_contours",
    "filter_contours_by_area",
    "filter_by_connectivity",
    "process_segmentation_map",
    "draw_class_contours",
]
