"""Centralised exception hierarchy for Zoya.

Design notes
------------
* One root error (``ZoyaError``) lets callers catch "anything Zoya raised"
  with a single ``except``.
* Sub-hierarchies mirror the subsystems so callers can be as specific as they
  like, e.g. ``except WindowNotFoundError`` vs. ``except AutomationError``.
* Tool-layer errors (validation / lookup) are separated from *execution*
  failures because their handling differs: a validation error is a caller bug,
  whereas an execution failure is usually a transient environment problem.
"""

from __future__ import annotations


class ZoyaError(Exception):
    """Base class for every error raised by Zoya."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigurationError(ZoyaError):
    """Raised when configuration is missing, malformed or invalid."""


# ---------------------------------------------------------------------------
# Automation subsystem
# ---------------------------------------------------------------------------
class AutomationError(ZoyaError):
    """Base class for all desktop-automation failures."""


class AutomationDisabledError(AutomationError):
    """Raised when an action is requested while automation is disabled."""


class InputSimulationError(AutomationError):
    """Keyboard / mouse input could not be simulated."""


class WindowNotFoundError(AutomationError):
    """No window matched the requested title/handle."""


class ProcessError(AutomationError):
    """Process launch / query / termination failed."""


class FileSystemError(AutomationError):
    """A file-system operation failed (missing path, permissions, ...)."""


class SystemControlError(AutomationError):
    """A system control (volume, brightness, power, ...) failed."""


# ---------------------------------------------------------------------------
# Tool / plugin layer
# ---------------------------------------------------------------------------
class ToolError(ZoyaError):
    """Base class for tool-layer problems (not execution crashes)."""


class ToolNotFoundError(ToolError):
    """The requested tool name is not registered."""


class ToolValidationError(ToolError):
    """The parameters supplied to a tool failed Pydantic validation."""


class ToolExecutionError(ToolError):
    """A tool completed but reported a domain-level failure in its result."""


__all__ = [
    "ZoyaError",
    "ConfigurationError",
    "AutomationError",
    "AutomationDisabledError",
    "InputSimulationError",
    "WindowNotFoundError",
    "ProcessError",
    "FileSystemError",
    "SystemControlError",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
]
