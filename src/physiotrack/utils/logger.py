"""File logging for physiotrack diagnostics."""

import logging

from .._logging import get_logger

__all__ = ["log_to_file"]


def log_to_file(log_file_path, level=logging.DEBUG, name="physiotrack"):
    """Also write physiotrack's log messages to a file.

    Adds a file handler to the package logger, so everything the library reports is
    captured on disk in addition to the console. Useful for long batch runs where the
    console scrollback is not enough.

    Calling this twice with the same path replaces the previous file handler rather than
    doubling every line.

    Args:
        log_file_path (str | os.PathLike): Destination log file. Opened in append mode.
        level (int, optional): Minimum level to record in the file. Defaults to
            ``logging.DEBUG``, which captures per-frame detail even when the console is
            at ``INFO``.
        name (str, optional): Logger to attach to. Defaults to the package logger,
            ``"physiotrack"``.

    Returns:
        logging.Logger: The logger the handler was attached to.

    Example:
        ```python
        import physiotrack as pt
        from physiotrack.utils import log_to_file

        log_to_file("run.log")
        pt.Video(source="clip.mp4").run("out.mp4")
        ```
    """
    target = get_logger(name)
    path = str(log_file_path)

    for handler in list(target.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == \
                logging.FileHandler(path, delay=True).baseFilename:
            target.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s %(funcName)s:%(lineno)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    target.addHandler(file_handler)
    # The logger's own level gates before handlers see the record, so make sure it is
    # permissive enough for the file level requested.
    if target.level > level:
        target.setLevel(level)
    return target
