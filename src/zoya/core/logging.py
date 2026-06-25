"""Logging setup for Zoya.

Uses the standard library ``logging`` package (no extra dependency) configured
with a console handler and a rotating file handler. Every module obtains its
logger via :func:`get_logger` so all log records share the ``zoya.`` namespace
and a single configuration point.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Single, project-wide log namespace.
_ROOT_LOGGER_NAME = "zoya"

# A readable single-line format: timestamp | LEVEL | logger | message.
_FORMAT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path = "logs",
    filename: str = "zoya.log",
) -> logging.Logger:
    """Configure the root ``zoya`` logger (idempotent — safe to call twice).

    Parameters
    ----------
    level:
        Logging level as a string ("DEBUG", "INFO", ...) or an int constant.
    log_dir:
        Directory for the rotating log file. Created if missing.
    filename:
        Name of the log file inside ``log_dir``.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level if isinstance(level, int) else str(level).upper())
    # Avoid duplicate handlers if setup_logging() is called again.
    logger.propagate = False
    if logger.handlers:
        return logger

    # Console -> stdout (so it shows in most launchers / IDEs).
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(_FORMAT)
    logger.addHandler(console)

    # Rotating file: 5 MB per file, keep 5 archives.
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_path / filename,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(_FORMAT)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``zoya`` namespace.

    Use a dotted hierarchical name, e.g. ``get_logger("automation.keyboard")``,
    which yields the logger ``zoya.automation.keyboard`` and inherits the
    configuration attached to the root ``zoya`` logger.
    """
    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


__all__ = ["setup_logging", "get_logger"]
