"""Explicit frame-orientation handling for video loading.

Phone cameras (notably iPhone) store the display rotation as *metadata* rather
than baking it into the pixels, so a frame decoded by OpenCV can come out sideways
or upside down. That metadata is unreliable across OpenCV/FFmpeg builds, so we do
not read it; instead the caller passes an explicit angle (0/90/180/270) when a clip
needs one, and we rotate the frames ourselves.

Use :func:`resolve_rotation` once to normalise the setting, then
:func:`apply_rotation` on each frame. Still images need no helper here: OpenCV's
``cv2.imread`` already honours EXIF orientation.
"""

import cv2

_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_rotation(frame, deg: int):
    """Rotate a frame clockwise by a fixed angle.

    Args:
        frame (np.ndarray): The BGR image/frame to rotate, shape ``(H, W, 3)``.
        deg (int): Clockwise rotation angle; one of ``90`` / ``180`` / ``270``.
            Any other value (including ``0``) returns the frame unchanged.

    Returns:
        np.ndarray: The rotated frame. For ``90``/``270`` the height and width are
            swapped; otherwise the shape is preserved.

    Example:
        ```python
        from physiotrack.capture.orientation import apply_rotation, resolve_rotation
        deg = resolve_rotation(90)
        rotated = apply_rotation(frame, deg)
        ```
    """
    return cv2.rotate(frame, _ROTATE[deg]) if deg in _ROTATE else frame


def resolve_rotation(rotate) -> int:
    """Normalise an explicit rotation setting to one of ``0``/``90``/``180``/``270``.

    There is no auto/metadata mode: either an explicit value is provided or no
    rotation is applied. Called once at setup so that [`apply_rotation`][physiotrack.capture.orientation.apply_rotation]
    can be applied per frame.

    Args:
        rotate: The requested rotation. ``None`` / ``"none"`` / ``0`` / ``"0"``
            mean no rotation; ``90`` / ``180`` / ``270`` (int or numeric string)
            select that clockwise angle. Values are taken modulo 360, and anything
            that is not one of the supported angles falls back to ``0``.

    Returns:
        int: A normalised angle, one of ``0``, ``90``, ``180`` or ``270``.

    Example:
        ```python
        from physiotrack.capture.orientation import resolve_rotation
        resolve_rotation("180")  # -> 180
        resolve_rotation(45)     # -> 0 (unsupported, falls back)
        ```
    """
    if rotate in (None, "none", 0, "0"):
        return 0
    try:
        deg = int(rotate) % 360
        return deg if deg in _ROTATE else 0
    except (TypeError, ValueError):
        return 0
