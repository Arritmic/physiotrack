"""Post-processing for dense class-index segmentation maps.

These helpers clean up the ``seg_map`` produced by the body-part and face-parsing
backends (see :class:`~physiotrack.Segmentation`): dropping contours outside a
region of interest, merging or removing classes, and suppressing speckle from
area or connectivity thresholds.

Every function takes a 2-D map of integer class indices where ``0`` is background,
and returns a new map of the same shape and dtype. Inputs are never modified in
place unless stated.
"""

import cv2
import numpy as np

__all__ = [
    "filter_by_box",
    "exclude_contours",
    "combine_contours",
    "remove_isolated_contours",
    "filter_contours_by_area",
    "filter_by_connectivity",
    "process_segmentation_map",
    "draw_class_contours",
]


def filter_by_box(segmentation_map, bbox=None):
    """Keep only contours fully contained in a bounding box.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices, ``0`` = background.
        bbox (np.ndarray | None): Shape ``(1, 4)`` as ``[[x1, y1, x2, y2]]``. When
            ``None`` the map is returned unchanged.

    Returns:
        np.ndarray: Map containing only the contours wholly inside ``bbox``.
    """
    if bbox is None:
        return segmentation_map

    filtered_map = np.zeros_like(segmentation_map, dtype=segmentation_map.dtype)
    x1, y1, x2, y2 = bbox[0]
    for cls in np.unique(segmentation_map):
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            is_within_bbox = all(
                (x1 <= point[0][0] <= x2) and (y1 <= point[0][1] <= y2) for point in contour
            )
            if is_within_bbox:
                cv2.drawContours(filtered_map, [contour], -1, int(cls), thickness=cv2.FILLED)
    return filtered_map


def exclude_contours(segmentation_map, exclude):
    """Erase the given classes from the map.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices. Modified in place.
        exclude (Iterable[int]): Class indices to set to background.

    Returns:
        np.ndarray: The map with the excluded classes set to ``0``.
    """
    for cls in exclude:
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(segmentation_map, [contour], -1, 0, thickness=cv2.FILLED)
    return segmentation_map


def combine_contours(segmentation_map, combine):
    """Relabel one class as another, merging their regions.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices. Modified in place.
        combine (dict[int, int]): Mapping of source class index to target class
            index. Keys and values may be strings; they are coerced to ``int``.

    Returns:
        np.ndarray: The map with each source class relabelled to its target.
    """
    for src_class, target_class in combine.items():
        src_class = int(src_class)
        target_class = int(target_class)
        combined_mask = np.zeros_like(segmentation_map, dtype=np.uint8)
        mask = np.uint8(segmentation_map == src_class) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(combined_mask, [contour], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(segmentation_map, [contour], -1, 0, thickness=cv2.FILLED)
        combined_contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in combined_contours:
            cv2.drawContours(segmentation_map, [contour], -1, target_class, thickness=cv2.FILLED)
    return segmentation_map


def remove_isolated_contours(segmentation_map):
    """Keep only the largest connected blob across all classes.

    Useful when the subject is the single foreground object and the backend has
    produced detached fragments elsewhere in the frame.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices.

    Returns:
        np.ndarray: Map restricted to the largest connected region, or an all-zero
            map when the input contains no foreground.
    """
    binary_map = np.uint8(segmentation_map > 0) * 255
    dilated_map = cv2.dilate(binary_map, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(segmentation_map)

    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_mask = np.zeros_like(binary_map)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return segmentation_map * (largest_contour_mask // 255)


def filter_contours_by_area(segmentation_map, min_area=0, max_area=None, exclude=None):
    """Drop contours whose pixel area falls outside a range.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices.
        min_area (float): Minimum contour area in pixels, inclusive.
        max_area (float | None): Maximum contour area in pixels, inclusive.
            ``None`` means unbounded.
        exclude (Iterable[int] | None): Class indices exempt from area filtering;
            their contours are always kept.

    Returns:
        np.ndarray: Map containing only the contours that passed the filter.
    """
    exclude = () if exclude is None else exclude
    filtered_map = np.zeros_like(segmentation_map, dtype=segmentation_map.dtype)

    for cls in np.unique(segmentation_map):
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cls in exclude:
                cv2.drawContours(filtered_map, [contour], -1, int(cls), thickness=cv2.FILLED)
                continue
            area = cv2.contourArea(contour)
            if area >= min_area and (max_area is None or area <= max_area):
                cv2.drawContours(filtered_map, [contour], -1, int(cls), thickness=cv2.FILLED)
    return filtered_map


def filter_by_connectivity(segmentation_map, connectivity_threshold=50):
    """Drop connected components smaller than a pixel-count threshold.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices.
        connectivity_threshold (int): Minimum number of pixels a component must
            contain to be kept. Components are found with 8-connectivity.

    Returns:
        np.ndarray: Map containing only components at or above the threshold.
    """
    filtered_map = np.zeros_like(segmentation_map, dtype=segmentation_map.dtype)

    for cls in np.unique(segmentation_map):
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):  # 0 is the background label
            if stats[i, cv2.CC_STAT_AREA] >= connectivity_threshold:
                filtered_map[labels == i] = cls
    return filtered_map


def process_segmentation_map(
    segmentation_map,
    exclude=None,
    combine=None,
    min_area=0,
    max_area=None,
    exclude_in_area_filtering=None,
    remove_unconnected=False,
    remove_isolated=False,
    connectivity_threshold=50,
):
    """Run the full clean-up chain over a segmentation map.

    Stages are applied in order: area filtering, connectivity filtering, isolated
    blob removal, class exclusion, then class merging. Each stage is skipped when
    its controlling argument leaves it disabled.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices.
        exclude (Iterable[int] | None): Class indices to erase.
        combine (dict[int, int] | None): Source-to-target class relabelling.
        min_area (float): Minimum contour area in pixels for area filtering.
        max_area (float | None): Maximum contour area in pixels, or ``None``.
        exclude_in_area_filtering (Iterable[int] | None): Classes exempt from area
            filtering.
        remove_unconnected (bool): Drop connected components below
            ``connectivity_threshold`` pixels.
        remove_isolated (bool): Keep only the largest connected blob.
        connectivity_threshold (int): Pixel threshold used when
            ``remove_unconnected`` is set.

    Returns:
        np.ndarray: The cleaned segmentation map.
    """
    if min_area > 0 or max_area is not None:
        segmentation_map = filter_contours_by_area(
            segmentation_map, min_area, max_area, exclude=exclude_in_area_filtering
        )
    if remove_unconnected:
        segmentation_map = filter_by_connectivity(segmentation_map, connectivity_threshold)
    if remove_isolated:
        segmentation_map = remove_isolated_contours(segmentation_map)
    if exclude:
        segmentation_map = exclude_contours(segmentation_map, exclude)
    if combine:
        segmentation_map = combine_contours(segmentation_map, combine)
    return segmentation_map


def draw_class_contours(segmentation_map, bbox=None):
    """Render labelled class contours onto a black canvas, for inspection.

    Each contour is outlined and annotated with ``"<class>:<index>"`` at its
    centroid, which makes it easy to read off the class indices to pass to
    :func:`process_segmentation_map`.

    Args:
        segmentation_map (np.ndarray): 2-D map of class indices.
        bbox (np.ndarray | None): Optional ``(1, 4)`` box drawn as
            ``[[x1, y1, x2, y2]]`` for reference.

    Returns:
        np.ndarray: A 3-channel BGR canvas the same height and width as the input.
    """
    canvas = cv2.cvtColor(np.zeros_like(segmentation_map, dtype=np.uint8), cv2.COLOR_GRAY2BGR)

    for cls in np.unique(segmentation_map):
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for idx, contour in enumerate(contours):
            cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 1)
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                cv2.putText(
                    canvas,
                    f"{int(cls)}:{idx}",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return canvas
