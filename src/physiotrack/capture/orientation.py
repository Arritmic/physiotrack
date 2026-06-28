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
    """Rotate a frame by ``deg`` (0/90/180/270); other values pass through."""
    return cv2.rotate(frame, _ROTATE[deg]) if deg in _ROTATE else frame


def resolve_rotation(rotate) -> int:
    """Normalise an explicit ``rotate`` setting to one of ``0/90/180/270``.

    ``rotate`` may be ``None``/``"none"``/``0`` (no rotation) or an explicit
    ``90/180/270``. There is no auto/metadata mode: either a value is provided or
    no rotation is applied. Unknown values fall back to ``0``.
    """
    if rotate in (None, "none", 0, "0"):
        return 0
    try:
        deg = int(rotate) % 360
        return deg if deg in _ROTATE else 0
    except (TypeError, ValueError):
        return 0
