"""Core cross-cutting concerns shared by every Zoya module.

This package is the *foundation layer*: configuration, logging, exceptions,
path resolution and application bootstrap. It is kept dependency-light so any
module (now and in the future) can import it without pulling in Windows-only
or heavy third-party packages.

Typical usage at an entry point::

    from zoya.core import bootstrap
    ctx = bootstrap()
    log = ctx.logger
    settings = ctx.settings
"""

from zoya.core.bootstrap import RuntimeContext, bootstrap, shutdown
from zoya.core.config import (
    AppConfig,
    AutomationSettings,
    Environment,
    HealthReport,
    LLMSettings,
    PathSettings,
    PROJECT_ROOT,
    SettingsManager,
    ZoyaSettings,
    get_settings_manager,
    load_settings,
    reset_settings_manager,
)
from zoya.core.exceptions import (
    AutomationDisabledError,
    AutomationError,
    BootstrapError,
    ConfigurationError,
    FileSystemError,
    InputSimulationError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    ProcessError,
    SystemControlError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    WindowNotFoundError,
    ZoyaError,
)
from zoya.core.logging import (
    HumanReadableFormatter,
    JsonFormatter,
    SecretRedactionFilter,
    clear_secrets,
    get_logger,
    register_secret,
    setup_logging,
    shutdown_logging,
)
from zoya.core.paths import PATHS, ProjectPaths

__all__ = [
    # bootstrap / lifecycle
    "RuntimeContext",
    "bootstrap",
    "shutdown",
    # config
    "AppConfig",
    "AutomationSettings",
    "Environment",
    "HealthReport",
    "LLMSettings",
    "PathSettings",
    "PROJECT_ROOT",
    "SettingsManager",
    "ZoyaSettings",
    "get_settings_manager",
    "load_settings",
    "reset_settings_manager",
    # paths
    "ProjectPaths",
    "PATHS",
    # logging
    "get_logger",
    "setup_logging",
    "shutdown_logging",
    "register_secret",
    "clear_secrets",
    "SecretRedactionFilter",
    "JsonFormatter",
    "HumanReadableFormatter",
    # exceptions — root
    "ZoyaError",
    "ConfigurationError",
    "BootstrapError",
    # exceptions — automation
    "AutomationError",
    "AutomationDisabledError",
    "InputSimulationError",
    "WindowNotFoundError",
    "ProcessError",
    "FileSystemError",
    "SystemControlError",
    # exceptions — tool layer
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    # exceptions — llm layer
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMTimeoutError",
]
