"""Opening a video writer that actually works.

OpenCV's ``VideoWriter`` does not raise when it cannot find an encoder: it returns an
object whose ``isOpened()`` is ``False``, and every ``write()`` call is silently
discarded. The run finishes "successfully" and leaves behind an unplayable, often
zero-byte file.

That is easy to hit in practice, because many ``opencv-python`` wheels ship without an
H.264 encoder: the OpenH264 runtime library is downloaded separately and, when it is
missing, FFmpeg logs ``Failed to load OpenH264 library`` and the writer never opens.

:func:`open_video_writer` tries H.264 first for broad playback support, verifies the
writer really opened, falls back to the always-available MPEG-4 encoder with a warning,
and raises if neither works — so a missing codec can never be mistaken for a successful
export.
"""

import warnings

import cv2

__all__ = ["open_video_writer"]

#: Encoders tried in order. H.264 first for compatibility, MPEG-4 as the fallback that
#: is compiled into every OpenCV build.
_CODECS = ("avc1", "mp4v")


def open_video_writer(path, fps, frame_size, codec=None):
    """Open a video writer, falling back if the preferred encoder is unavailable.

    Args:
        path (str | os.PathLike): Destination video file.
        fps (float): Frame rate to record in the container. Must be > 0.
        frame_size (tuple[int, int]): ``(width, height)`` in pixels.
        codec (str, optional): Force a single FourCC (e.g. ``"mp4v"``, ``"XVID"``)
            instead of trying H.264 then MPEG-4. Defaults to ``None``.

    Returns:
        cv2.VideoWriter: An opened writer. Release it when done.

    Raises:
        ValueError: If ``fps`` or ``frame_size`` is not usable.
        RuntimeError: If no encoder could open the file. The message names what was
            tried, since the cause is almost always a missing OpenH264 runtime.

    Warns:
        RuntimeWarning: If H.264 was unavailable and a fallback encoder was used, so the
            output format is not silently different from what was asked for.

    Example:
        ```python
        from physiotrack.capture import open_video_writer

        writer = open_video_writer("out.mp4", 30.0, (1920, 1080))
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        ```
    """
    width, height = (int(v) for v in frame_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size!r}.")
    fps = float(fps)
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps!r}.")

    candidates = (codec,) if codec else _CODECS
    for name in candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*name),
                                 fps, (width, height))
        if writer.isOpened():
            if name != candidates[0]:
                warnings.warn(
                    f"The {candidates[0]!r} encoder is unavailable in this OpenCV build; "
                    f"wrote {name!r} instead. For H.264 output, install the OpenH264 "
                    f"runtime library.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return writer
        writer.release()

    raise RuntimeError(
        f"Could not open a video writer for {str(path)!r} at {width}x{height} @ {fps} fps. "
        f"Tried {', '.join(repr(c) for c in candidates)}. This usually means the OpenH264 "
        f"runtime library is missing and no fallback encoder is compiled into this OpenCV "
        f"build."
    )
