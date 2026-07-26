"""Filesystem locations used by the library.

Model weights are large and shared between projects, so they belong in a per-user
cache rather than inside the installed package. Writing into ``site-packages``
breaks read-only and containerised installs, defeats Docker layer caching, and
prevents two environments from sharing one download.

Resolution order for the cache root:

1. ``$PHYSIOTRACK_HOME`` when set — the explicit override.
2. ``$XDG_CACHE_HOME/physiotrack`` when set (Linux convention, honoured anywhere).
3. The platform default: ``%LOCALAPPDATA%\\physiotrack`` on Windows,
   ``~/Library/Caches/physiotrack`` on macOS, ``~/.cache/physiotrack`` elsewhere.
"""

import os
import shutil
import sys
from pathlib import Path

from ._logging import get_logger

logger = get_logger(__name__)

__all__ = ["cache_root", "weights_dir", "legacy_weights_dir", "migrate_weight_cache"]


def cache_root() -> Path:
    """Return the root directory for cached downloads.

    Returns:
        pathlib.Path: The cache root. The directory is not created here; callers
        that write into it are responsible for creating what they need.
    """
    override = os.environ.get("PHYSIOTRACK_HOME")
    if override:
        return Path(override).expanduser()

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "physiotrack"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or "~/AppData/Local"
        return Path(base).expanduser() / "physiotrack"
    if sys.platform == "darwin":
        return Path("~/Library/Caches/physiotrack").expanduser()
    return Path("~/.cache/physiotrack").expanduser()


def weights_dir() -> Path:
    """Return the directory holding cached model weights, creating it if needed.

    Returns:
        pathlib.Path: ``<cache_root>/weights``, guaranteed to exist.
    """
    path = cache_root() / "weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_weights_dir() -> Path:
    """Return the pre-1.1 in-package weights directory.

    Releases before 1.1 downloaded checkpoints into ``physiotrack/modules/model_data``
    inside the installed package. Nothing reads from there any more — this exists only
    so [`migrate_weight_cache`][physiotrack.migrate_weight_cache] can find files already
    on disk and so the library can point at them instead of silently re-downloading
    several gigabytes.

    Returns:
        pathlib.Path: The legacy directory, whether or not it exists.
    """
    return Path(__file__).parent / "modules" / "model_data"


def migrate_weight_cache(dry_run: bool = False) -> list:
    """Move checkpoints from the pre-1.1 in-package directory into the user cache.

    Run this once after upgrading if an earlier version already downloaded weights.
    Files are moved, not copied, so the multi-gigabyte payload is not duplicated;
    anything already present in the cache is left alone and the legacy copy removed.

    Args:
        dry_run (bool): Report what would move without touching the filesystem.
            Defaults to ``False``.

    Returns:
        list[tuple[pathlib.Path, pathlib.Path]]: The ``(source, destination)`` pairs
            that were moved (or, with ``dry_run``, would be).

    Example:
        ```python
        from physiotrack import migrate_weight_cache
        migrate_weight_cache(dry_run=True)   # see what would move
        migrate_weight_cache()               # do it
        ```
    """
    legacy = legacy_weights_dir()
    if not legacy.is_dir():
        logger.info("No legacy weight directory at %s; nothing to migrate.", legacy)
        return []

    target = weights_dir()
    moved = []
    for src in sorted(p for p in legacy.rglob("*") if p.is_file()):
        dst = target / src.relative_to(legacy)
        if dst.exists():
            if not dry_run:
                src.unlink()  # already cached; drop the stale in-package duplicate
            continue
        moved.append((src, dst))
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    total_gb = sum(s.stat().st_size for s, _ in moved) / 1e9 if dry_run else 0.0
    if dry_run:
        logger.info("Would move %d file(s) (%.2f GB) from %s to %s",
                    len(moved), total_gb, legacy, target)
    else:
        logger.info("Moved %d file(s) from %s to %s", len(moved), legacy, target)
    return moved
