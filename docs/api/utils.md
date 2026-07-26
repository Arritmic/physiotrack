# Utilities

Shared helpers: the weight cache, file logging, display sizing, the floor-plane
homography behind the bird's-eye [`RadarView`][physiotrack.core.RadarView], and
post-processing for dense segmentation maps.

## Weight cache

Weights are cached outside the installed package — see
[Where weights are cached](../model-zoo.md#where-weights-are-cached) for the resolution
order and how to share one cache between environments.

::: physiotrack.migrate_weight_cache

::: physiotrack._paths.weights_dir

::: physiotrack._paths.cache_root

## Logging

::: physiotrack.set_log_level

::: physiotrack.utils.log_to_file

## Display

::: physiotrack.utils.get_screen_size

::: physiotrack.utils.resize_frame_for_display

## Spatial transforms

::: physiotrack.utils.compute_homography

::: physiotrack.utils.transform_point

::: physiotrack.utils.get_foot_position

## Segmentation-map post-processing

::: physiotrack.utils.process_segmentation_map

::: physiotrack.utils.filter_by_box

::: physiotrack.utils.filter_contours_by_area

::: physiotrack.utils.filter_by_connectivity

::: physiotrack.utils.remove_isolated_contours

::: physiotrack.utils.exclude_contours

::: physiotrack.utils.combine_contours

::: physiotrack.utils.draw_class_contours
