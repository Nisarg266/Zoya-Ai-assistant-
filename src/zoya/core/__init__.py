"""Core cross-cutting concerns shared by every Zoya module.

Kept dependency-light so any module (now and in the future) can import it
without pulling in Windows-only or heavy third-party packages.
"""

from zoya.core.config import (
    AppConfig,
    AutomationSettings,
    PathSettings,
    ZoyaSettings,
    load_settings,
)
from zoya.core.exceptions import (
    AutomationDisabledError,
    AutomationError,
    ConfigurationError,
    FileSystemError,
    InputSimulationError,
    ProcessError,
    SystemControlError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    WindowNotFoundError,
    ZoyaError,
)
from zoya.core.logging import get_logger, setup_logging

__all__ = [
    # config
    "AppConfig",
    "AutomationSettings",
    "PathSettings",
    "ZoyaSettings",
    "load_settings",
    # logging
    "get_logger",
    "setup_logging",
    # exceptions
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
    "ToolExecutionError",
    "ToolValidationError",
]
