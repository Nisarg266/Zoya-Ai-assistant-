"""The composite settings object that the whole application consumes.

:class:`ZoyaSettings` fuses the three configuration sources into one immutable,
fully-validated value:

    .env  ──►  AppConfig            (secrets, flags, environment)
    YAML ──►  AutomationSettings   (tunable automation defaults)
              PathSettings          (default on-disk locations)
              LLMSettings           (generation + ReAct tunables)

Downstream code only ever depends on :class:`ZoyaSettings`, never on the
individual loaders — that single seam keeps the rest of Zoya decoupled from the
*how* of configuration loading.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator

from zoya.core.config.env import AppConfig
from zoya.core.config.yaml_settings import (
    AutomationSettings,
    LLMSettings,
    PathSettings,
)
from zoya.core.paths import PROJECT_ROOT


class ZoyaSettings(BaseModel):
    """The full, validated configuration consumed by the application."""

    app: AppConfig
    automation: AutomationSettings = AutomationSettings()
    paths: PathSettings = PathSettings()
    llm: LLMSettings = LLMSettings()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        """Absolute path to the project root."""
        return PROJECT_ROOT

    @property
    def log_path(self) -> Path:
        """Resolved directory for log files (created lazily elsewhere)."""
        return (PROJECT_ROOT / self.app.log_dir).resolve()

    @property
    def is_production(self) -> bool:
        return self.app.environment.is_production

    @property
    def is_development(self) -> bool:
        return self.app.environment.is_development

    @property
    def dry_run(self) -> bool:
        """Shortcut for the automation dry-run flag."""
        return self.app.automation_dry_run

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _validate_log_level_known(self) -> "ZoyaSettings":
        """Reject an unrecognised log level early instead of silently defaulting."""
        import logging as _stdlogging

        level = self.app.log_level
        # getattr returns the int level for known names, else the original string.
        resolved = getattr(_stdlogging, level, None)
        if not isinstance(resolved, int):
            raise ValueError(
                f"Unknown LOG_LEVEL {level!r}. Use one of: "
                "DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )
        return self


__all__ = ["ZoyaSettings"]
