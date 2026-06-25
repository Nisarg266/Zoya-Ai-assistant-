"""Environment-variable / ``.env`` loader.

This is the *secrets & flags* layer of Zoya's configuration: values that change
between machines (API keys, environment selection, feature toggles) and that
must **never** be committed to version control.

It is implemented with :mod:`pydantic_settings`, which gives us:

* automatic reading from a ``.env`` file **and** the live OS environment,
* case-insensitive keys,
* type coercion + validation,
* a documented default for every field.

The live environment always wins over the file, which is the expected
12-factor behaviour.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zoya.core.paths import PATHS


class Environment(str, Enum):
    """The runtime environment Zoya is executing in.

    The value influences how strict validation is — see
    :mod:`zoya.core.bootstrap`.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self is Environment.DEVELOPMENT


class AppConfig(BaseSettings):
    """Top-level application settings loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(PATHS.env_file),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- General application --------------------------------------------
    app_name: str = "Zoya"
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_dir: str = "logs"
    config_path: str = "config/settings.yaml"

    # --- Automation feature flags ---------------------------------------
    #: Master kill-switch. When ``False`` the automation facade refuses to run.
    automation_enabled: bool = True
    #: Dry-run mode: no real input is sent; tools only log what they *would* do.
    automation_dry_run: bool = False
    #: Reserved for the future mouse corner-abort failsafe.
    automation_failsafe: bool = True

    # --- LLM settings ---------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    #: API endpoint version. v1beta is the stable endpoint for gemini-2.5-pro.
    gemini_api_version: str = "v1beta"

    # --- LLM transport / resilience -------------------------------------
    #: Per-request timeout (seconds) for a single Gemini API call.
    llm_timeout: float = Field(120.0, gt=0)
    #: Max attempts (including the first) for retryable errors (429 / 5xx).
    llm_retry_attempts: int = Field(4, ge=1)
    #: Base delay (seconds) for the first retry backoff.
    llm_retry_base_delay: float = Field(1.0, ge=0)
    #: Upper bound (seconds) on any single retry delay.
    llm_retry_max_delay: float = Field(30.0, ge=0)

    # --- validators -----------------------------------------------------
    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> str:
        """Accept any casing (``"debug"``, ``"Debug"`` ...) and normalise to
        upper-case so it maps cleanly to :mod:`logging` constants."""
        return str(value).strip().upper()

    @field_validator("environment", mode="before")
    @classmethod
    def _normalise_environment(cls, value: object) -> object:
        """Be lenient about casing for environment names."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    # --- convenience ----------------------------------------------------
    @property
    def has_api_key(self) -> bool:
        """``True`` when a non-empty Gemini API key is configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


__all__ = ["Environment", "AppConfig"]
