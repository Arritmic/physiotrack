"""Adapter between the predictor result objects and the signals functions.

The predictors return [`Result`][physiotrack.Result] objects whose instances hold
[`Keypoints`][physiotrack.Keypoints]; the analysis functions in this subpackage were
written against the serialized ``{"id", "x", "y", "confidence"}`` dict form that
``Result.to_dict()`` and the ``Video`` JSON output produce.

This module lets both be passed interchangeably, so callers no longer have to reach
for ``result.to_dict()["instances"][0]["keypoints"]`` and discard the object they
were just handed.
"""

from typing import Any, Iterable, List

__all__ = ["as_keypoint_dicts", "as_frame_records"]


def as_frame_records(data: Any) -> List[dict]:
    """Normalise a video run's output to the per-frame record list.

    The sequence functions in this subpackage (``extract_keypoints_sequence`` and
    friends) read per-frame dicts. [`Video.run`][physiotrack.Video.run] returns a
    [`VideoResults`][physiotrack.VideoResults] of
    [`FrameResult`][physiotrack.FrameResult] objects, and a JSON file loaded from disk is
    already the dict form — this accepts either, so callers do not have to convert.

    Args:
        data: A ``VideoResults``, any iterable of ``FrameResult``, or a list of frame
            dicts as produced by
            [`VideoResults.to_dict_list`][physiotrack.VideoResults.to_dict_list].

    Returns:
        list[dict]: Frame records, each with ``frame_id``, ``timestamp`` and
            ``instances``.

    Raises:
        TypeError: If ``data`` is not iterable.

    Example:
        ```python
        import physiotrack as pt
        from physiotrack.signals import extract_keypoints_sequence

        results = pt.Video(source="clip.mp4", pose=pt.Pose.Person()).run()
        df = extract_keypoints_sequence(results)      # accepts the objects directly
        ```
    """
    if data is None:
        return []
    if not isinstance(data, Iterable):
        raise TypeError(
            f"Expected VideoResults, an iterable of FrameResult, or a list of frame "
            f"dicts; got {type(data).__name__}."
        )
    out = []
    for frame in data:
        if isinstance(frame, dict):
            out.append(frame)
        elif hasattr(frame, "to_dict"):
            out.append(frame.to_dict())
        else:
            raise TypeError(
                f"Frame records must be dicts or FrameResult objects; got "
                f"{type(frame).__name__}."
            )
    return out


def as_keypoint_dicts(source: Any) -> List[dict]:
    """Normalise any supported keypoint container to a list of keypoint dicts.

    Accepts, in order of preference:

    - [`Keypoints`][physiotrack.Keypoints] — one subject's landmarks.
    - [`Instance`][physiotrack.Instance] — its ``.keypoints`` are used.
    - [`Result`][physiotrack.Result] — only when it holds exactly one instance, since
      a per-frame measurement is defined for a single subject. Index the result
      (``result[0]``) to choose explicitly when there are several.
    - ``list[dict]`` — the serialized form, returned unchanged.

    Args:
        source: A ``Keypoints``, ``Instance``, single-instance ``Result``, or a list of
            ``{"id", "x", "y", "confidence"}`` dicts.

    Returns:
        list[dict]: Keypoint dicts with ``id``, ``x``, ``y``, ``confidence`` and, when
            the source carries it, ``z``. Empty when the source has no keypoints.

    Raises:
        TypeError: If ``source`` is not one of the supported types.
        ValueError: If a ``Result`` holds more than one instance, which would make the
            choice of subject implicit.

    Example:
        ```python
        import physiotrack as pt
        from physiotrack.signals import joint_angles

        result = pt.Pose.Person().predict(frame)
        angles = joint_angles(result[0])          # an Instance
        angles = joint_angles(result[0].keypoints)  # or its Keypoints
        ```
    """
    if source is None:
        return []

    # Already the serialized form (or empty).
    if isinstance(source, (list, tuple)):
        return list(source)

    # The serialized forms the library itself writes to JSON: an instance dict carrying
    # "keypoints", or a frame record carrying "instances". Accepting these means a
    # round-tripped JSON is as usable as a live result object.
    if isinstance(source, dict):
        if "keypoints" in source:
            return list(source["keypoints"] or [])
        if "instances" in source:
            instances = source["instances"] or []
            if not instances:
                return []
            if len(instances) > 1:
                raise ValueError(
                    f"This measurement is defined for one subject, but the record holds "
                    f"{len(instances)} instances. Pass a single instance to say which "
                    f"subject you mean."
                )
            return as_keypoint_dicts(instances[0])
        raise TypeError(
            "A dict input must carry either 'keypoints' (an instance) or 'instances' "
            f"(a frame record); got keys {sorted(source)}."
        )

    # Duck-typed to avoid importing physiotrack.results here, which would pull the
    # result objects into the pure-analysis import path.
    #
    # Order matters: a Result exposes *both* `.instances` and a `.keypoints` property
    # (the latter being one collection per instance), so it must be recognised first.
    instances = getattr(source, "instances", None)
    if instances is not None:
        if len(instances) == 0:
            return []
        if len(instances) > 1:
            raise ValueError(
                f"This measurement is defined for one subject, but the result holds "
                f"{len(instances)} instances. Pass a single instance, e.g. "
                f"`result[0]`, to say which subject you mean."
            )
        keypoints = getattr(instances[0], "keypoints", None)
    else:
        # An Instance exposes `.keypoints`; a Keypoints collection is iterable itself.
        keypoints = getattr(source, "keypoints", None)
        if keypoints is None:
            keypoints = source

    if not hasattr(keypoints, "__iter__"):
        raise TypeError(
            f"Expected Keypoints, an Instance, a single-instance Result, or a list of "
            f"keypoint dicts; got {type(source).__name__}."
        )

    out = []
    for keypoint in keypoints:
        if isinstance(keypoint, dict):
            out.append(keypoint)
            continue
        entry = {
            "id": keypoint.id,
            "x": keypoint.x,
            "y": keypoint.y,
            "confidence": keypoint.confidence,
        }
        z = getattr(keypoint, "z", None)
        if z is not None:
            entry["z"] = z
        out.append(entry)
    return out
