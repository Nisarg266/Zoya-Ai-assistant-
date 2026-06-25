"""Configuration loading for Zoya.

Two complementary sources are merged into a single validated object:

1. **Environment / .env** → :class:`AppConfig` (pydantic-settings). This holds
   secrets, environment selection and feature flags — values that change between
   machines and should NOT live in version control.
2. **YAML** (``config/settings.yaml``) → :class:`AutomationSettings` and
   :class:`PathSettings`. This holds tunable defaults that are the same across
   machines and are meant to be edited/committed by the developer.

The final :class:`ZoyaSettings` composes all three. Use :func:`load_settings`
(cached) to obtain it anywhere in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root from this file's location:
#   src/zoya/core/config.py  ->  parents[3] == project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1) Environment-driven configuration (.env)
# ---------------------------------------------------------------------------
class AppConfig(BaseSettings):
    """Top-level application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Zoya"
    environment: str = "development"
    log_level: str = "INFO"
    log_dir: str = "logs"
    config_path: str = "config/settings.yaml"

    # Automation feature flags.
    automation_enabled: bool = True
    automation_dry_run: bool = False
    automation_failsafe: bool = True

    # LLM settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# 2) YAML-driven configuration
# ---------------------------------------------------------------------------
class AutomationSettings(BaseModel):
    """Tunable automation defaults (loaded from settings.yaml)."""

    default_type_interval: float = Field(0.0, ge=0, description="Seconds between characters when typing.")
    key_press_interval: float = Field(0.1, ge=0, description="Seconds between repeated key taps.")
    mouse_move_duration: float = Field(0.3, ge=0, description="Seconds for a smooth cursor move.")
    mouse_move_steps: int = Field(50, ge=1, description="Interpolation steps per smooth move.")
    click_interval: float = Field(0.1, ge=0, description="Seconds between successive clicks.")
    scroll_amount: int = Field(3, ge=1, description="Default scroll magnitude.")
    launch_timeout: float = Field(10.0, ge=0, description="Seconds to wait when launching an app.")
    screenshot_dir: str = "screenshots"
    safe_delete: bool = True


class PathSettings(BaseModel):
    """Default on-disk locations used by Zoya."""

    notes_dir: str = "data/notes"


# ---------------------------------------------------------------------------
# 3) Composite settings object
# ---------------------------------------------------------------------------
class ZoyaSettings(BaseModel):
    """The full, validated configuration consumed by the application."""

    app: AppConfig
    automation: AutomationSettings = AutomationSettings()
    paths: PathSettings = PathSettings()

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def log_path(self) -> Path:
        return PROJECT_ROOT / self.app.log_dir


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    """Return YAML contents as a dict, or {} if the file is absent/empty."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


@lru_cache(maxsize=1)
def load_settings() -> ZoyaSettings:
    """Build and cache the global :class:`ZoyaSettings` instance.

    Cached so repeated calls are free; tests that need a fresh object can call
    ``load_settings.cache_clear()``.
    """
    app = AppConfig()

    yaml_path = (PROJECT_ROOT / app.config_path).resolve()
    raw = _load_yaml(yaml_path)

    automation = AutomationSettings(**raw.get("automation", {}))
    paths = PathSettings(**raw.get("paths", {}))

    return ZoyaSettings(app=app, automation=automation, paths=paths)


__all__ = [
    "PROJECT_ROOT",
    "AppConfig",
    "AutomationSettings",
    "PathSettings",
    "ZoyaSettings",
    "load_settings",
]
