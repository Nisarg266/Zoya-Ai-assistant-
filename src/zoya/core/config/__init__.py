"""Configuration package for Zoya.

Public surface — import anything you need from ``zoya.core.config``:

    from zoya.core.config import ZoyaSettings, AppConfig, load_settings

Internally the work is split across focused modules:

* :mod:`zoya.core.config.env`           — ``.env`` / environment loader
* :mod:`zoya.core.config.yaml_settings` — YAML tunable defaults
* :mod:`zoya.core.config.models`        — composite ``ZoyaSettings``
* :mod:`zoya.core.config.manager`       — :class:`SettingsManager` lifecycle
"""

from __future__ import annotations

from zoya.core.config.env import AppConfig, Environment
from zoya.core.config.manager import (
    HealthReport,
    SettingsManager,
    get_settings_manager,
    load_settings,
    reset_settings_manager,
)
from zoya.core.config.models import ZoyaSettings
from zoya.core.config.yaml_settings import (
    AutomationSettings,
    LLMSettings,
    PathSettings,
)
from zoya.core.paths import PROJECT_ROOT

__all__ = [
    # root path (kept for backwards compatibility)
    "PROJECT_ROOT",
    # env layer
    "AppConfig",
    "Environment",
    # yaml layer
    "AutomationSettings",
    "PathSettings",
    "LLMSettings",
    # composite
    "ZoyaSettings",
    # manager
    "SettingsManager",
    "HealthReport",
    "get_settings_manager",
    "load_settings",
    "reset_settings_manager",
]
