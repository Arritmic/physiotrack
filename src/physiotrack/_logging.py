"""Logging setup for the library.

All diagnostics go through the ``physiotrack`` logger rather than ``print``, so they
can be filtered, redirected, captured in tests, or silenced entirely. A stream handler
is attached at ``INFO`` by default so scripts still show progress out of the box; call
:func:`set_log_level` or reconfigure the ``physiotrack`` logger to change that.

Two conventions apply across the package:

- A predictor's ``verbose`` flag chooses the *level* its own messages are emitted at —
  ``INFO`` when verbose, ``DEBUG`` when not. It never reconfigures global logging, so
  one noisy object cannot silence another.
- Conditions the caller should act on (a missing codec, an unusable configuration) are
  raised as exceptions or issued with :mod:`warnings`, not logged. Logging is for
  progress; warnings are for problems.
"""

import logging
import sys

__all__ = ["get_logger", "set_log_level", "logger"]

_ROOT_NAME = "physiotrack"

#: The package-wide logger. Child loggers are created per module by :func:`get_logger`.
logger = logging.getLogger(_ROOT_NAME)


def _configure() -> None:
    """Attach a stream handler once, so importing the library twice is harmless."""
    if getattr(logger, "_physiotrack_configured", False):
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Diagnostics are the library's own; do not also hand them to the root logger,
    # which would duplicate every line in applications that configure logging.
    logger.propagate = False
    logger._physiotrack_configured = True


_configure()


def get_logger(name: str) -> logging.Logger:
    """Return the module-scoped child logger.

    Args:
        name (str): Usually ``__name__``. A ``physiotrack.*`` name is used as-is so the
            logger hierarchy mirrors the package layout.

    Returns:
        logging.Logger: A child of the ``physiotrack`` logger, so it inherits the level
            and handler configured here.

    Example:
        ```python
        from physiotrack._logging import get_logger

        logger = get_logger(__name__)
        logger.info("loaded %s", model_name)
        ```
    """
    if name == _ROOT_NAME or name.startswith(f"{_ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def set_log_level(level) -> None:
    """Set the verbosity of every physiotrack logger.

    Args:
        level (int | str): A :mod:`logging` level, either the constant
            (``logging.DEBUG``) or its name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``). Use ``"WARNING"`` to silence progress messages while keeping
            problems visible, or ``"DEBUG"`` to see per-frame detail.

    Example:
        ```python
        import physiotrack as pt
        pt.set_log_level("WARNING")   # quiet
        pt.set_log_level("DEBUG")     # everything
        ```
    """
    logger.setLevel(level)
