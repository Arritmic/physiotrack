"""Video capture and frame orientation.

Holds the [`Video`][physiotrack.Video] orchestrator and the frame-rotation helpers
behind its ``orient=`` argument. The rotation helpers are exported here so callers
writing their own capture loop — as the vitals example does — can reuse the same
orientation handling as the orchestrator instead of reaching into a private module.
"""

from .orientation import apply_rotation, resolve_rotation
from .writer import open_video_writer
from .video import Video

__all__ = ["Video", "apply_rotation", "resolve_rotation", "open_video_writer"]
