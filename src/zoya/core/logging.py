"""Production-grade logging for Zoya.

Built on the standard-library :mod:`logging` package — no extra dependency —
and configured through a single entry point (:func:`setup_logging`) so the whole
codebase shares the ``zoya.`` namespace and one set of handlers.

Features
--------
* **Console handler** with optional colourised levels (ANSI; auto-disabled when
  the stream is not a TTY, so redirected output stays clean).
* **Rotating file handler** (5 MB / 5 archives) so logs never grow unbounded.
* **Secret redaction filter** — any record that stringifies an API key is
  scrubbed before it reaches a handler. Register extra secret substrings with
  :func:`register_secret`.
* **Optional JSON formatter** for structured ingestion (set ``json=True``).
* **Idempotent** — calling :func:`setup_logging` twice never duplicates
  handlers; :func:`shutdown_logging` tears everything down cleanly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable

# Single, project-wide log namespace.
_ROOT_LOGGER_NAME = "zoya"

#: Placeholder shown in place of a redacted secret.
_REDACTED = "***REDACTED***"


# ---------------------------------------------------------------------------
# Secret redaction registry
# ---------------------------------------------------------------------------
class _SecretRegistry:
    """Holds the set of secret substrings to scrub from log records."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def add(self, value: str | None) -> None:
        if value and len(value) >= 4:  # ignore trivially short/noisy values
            self._secrets.add(value)

    def add_many(self, values: Iterable[str | None]) -> None:
        for value in values:
            self.add(value)

    @property
    def secrets(self) -> frozenset[str]:
        return frozenset(self._secrets)

    def clear(self) -> None:
        self._secrets.clear()


_secret_registry = _SecretRegistry()


def register_secret(value: str | None) -> None:
    """Register a secret (e.g. an API key) so it is redacted from all logs.

    Values shorter than 4 characters are ignored to avoid masking common words.
    """
    _secret_registry.add(value)


def clear_secrets() -> None:
    """Forget all registered secrets (useful in tests)."""
    _secret_registry.clear()


# ---------------------------------------------------------------------------
# Filters & formatters
# ---------------------------------------------------------------------------
class SecretRedactionFilter(logging.Filter):
    """Scrub registered secret substrings from a log record's rendered text.

    Runs on every record emitted by the ``zoya`` logger. We redact
    ``record.getMessage()`` *and* mutate ``record.args`` so downstream handlers
    that re-render (rare) also stay safe.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = _secret_registry.secrets
        if not secrets:
            return True

        message = record.getMessage()
        changed = message
        for secret in secrets:
            if secret and secret in changed:
                changed = changed.replace(secret, _REDACTED)

        if changed != message:
            # Freeze the already-rendered, scrubbed text and drop the args so
            # no handler tries to re-interpolate the original message.
            record.msg = changed
            record.args = None
        return True


# ---------------------------------------------------------------------------
# ANSI colour support (optional, TTY-only)
# ---------------------------------------------------------------------------
class _Color:
    """Minimal ANSI colour table. Empty strings when colour is disabled."""

    RESET = "\033[0m"
    _TABLE = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[35m",  # magenta
    }

    @classmethod
    def wrap(cls, level: int, text: str, *, enabled: bool) -> str:
        if not enabled:
            return text
        code = cls._TABLE.get(level)
        return f"{code}{text}{cls.RESET}" if code else text


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
_HUMAN_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class HumanReadableFormatter(logging.Formatter):
    """Single-line, readable formatter used by the console (and file)."""

    def __init__(self, *, color: bool = False) -> None:
        super().__init__(fmt=_HUMAN_FORMAT, datefmt=_DATE_FORMAT)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        # Colour only the level name to keep the rest readable.
        if self._color:
            record.levelname = _Color.wrap(
                record.levelno, record.levelname, enabled=True
            )
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """One JSON object per line — handy for structured log aggregation."""

    _EXTRA_BLOCKLIST = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any custom/extra fields the caller passed to the log call.
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._EXTRA_BLOCKLIST and not k.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = getattr(logging, str(level).strip().upper(), None)
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown log level: {level!r}")
    return resolved


def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path = "logs",
    filename: str = "zoya.log",
    *,
    json_logs: bool = False,
    color_console: bool | None = None,
    enable_file: bool = True,
) -> logging.Logger:
    """Configure the root ``zoya`` logger (idempotent — safe to call twice).

    Parameters
    ----------
    level:
        Logging level as a string (``"DEBUG"``, ``"INFO"``, ...) or an int.
    log_dir:
        Directory for the rotating log file. Created if missing.
    filename:
        Name of the log file inside ``log_dir``.
    json_logs:
        Emit JSON-formatted records (great for ingestion). Defaults to
        human-readable lines.
    color_console:
        Colourise the level name on the console. ``None`` auto-detects: on if
        the stream is a TTY, off otherwise.
    enable_file:
        Set ``False`` to skip the file handler (useful for unit tests / CI).
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(_coerce_level(level))
    logger.propagate = False

    # Always (re)apply the secret-redaction filter, even if handlers exist.
    for handler in logger.handlers:
        if not any(isinstance(f, SecretRedactionFilter) for f in handler.filters):
            handler.addFilter(SecretRedactionFilter())

    if logger.handlers:
        # Already configured — respect the new level and stop.
        return logger

    # ---- console handler ------------------------------------------------
    use_color = sys.stdout.isatty() if color_console is None else color_console
    console = logging.StreamHandler(stream=sys.stdout)
    if json_logs:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(HumanReadableFormatter(color=use_color))
    console.addFilter(SecretRedactionFilter())
    logger.addHandler(console)

    # ---- rotating file handler -----------------------------------------
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_path / filename,
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=5,
            encoding="utf-8",
        )
        # Files are always plain text (JSON optional but not coloured).
        file_handler.setFormatter(
            JsonFormatter() if json_logs else HumanReadableFormatter(color=False)
        )
        file_handler.addFilter(SecretRedactionFilter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``zoya`` namespace.

    Pass a dotted hierarchical name, e.g. ``get_logger("automation.keyboard")``,
    which yields the logger ``zoya.automation.keyboard`` and inherits the
    configuration attached to the root ``zoya`` logger.
    """
    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def shutdown_logging() -> None:
    """Flush + remove every handler on the ``zoya`` logger.

    Use this in a ``finally`` block at process exit or to fully reset logging
    between tests.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


__all__ = [
    "setup_logging",
    "get_logger",
    "shutdown_logging",
    "register_secret",
    "clear_secrets",
    "SecretRedactionFilter",
    "JsonFormatter",
    "HumanReadableFormatter",
]
