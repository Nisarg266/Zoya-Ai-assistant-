"""The :class:`SettingsManager` — lifecycle owner for Zoya's configuration.

Responsibilities
----------------
* **Load** configuration from the two sources (``.env`` + YAML) into a single
  validated :class:`ZoyaSettings`.
* **Cache** the result so repeated lookups are free, with explicit
  :meth:`reload` to pick up runtime changes (e.g. after the user edits the
  YAML while Zoya is running).
* **Validate** cross-cutting invariants that pydantic alone can't express
  (e.g. "production *requires* an API key").
* **Sanity-check** the environment, returning structured :class:`HealthReport`
  data the bootstrap layer can log or act on.

The manager is a *singleton-ish* helper: :func:`get_settings_manager` returns
one shared instance, but the class is fully instantiable on its own for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from zoya.core.config.env import AppConfig
from zoya.core.config.models import ZoyaSettings
from zoya.core.config.yaml_settings import (
    AutomationSettings,
    LLMSettings,
    PathSettings,
)
from zoya.core.exceptions import ConfigurationError
from zoya.core.paths import PATHS


# ---------------------------------------------------------------------------
# Health report
# ---------------------------------------------------------------------------
@dataclass
class HealthReport:
    """Result of a configuration sanity check.

    ``issues`` is a list of ``("error"|"warning", message)`` tuples so callers
    can decide whether to abort (any error) or just log warnings.
    """

    issues: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when there are no *errors* (warnings are tolerated)."""
        return not any(level == "error" for level, _ in self.issues)

    @property
    def errors(self) -> list[str]:
        return [msg for level, msg in self.issues if level == "error"]

    @property
    def warnings(self) -> list[str]:
        return [msg for level, msg in self.issues if level == "warning"]

    def add_error(self, message: str) -> None:
        self.issues.append(("error", message))

    def add_warning(self, message: str) -> None:
        self.issues.append(("warning", message))


# ---------------------------------------------------------------------------
# YAML loading helper
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    """Return YAML contents as a dict, or ``{}`` if the file is absent/empty.

    Raises :class:`ConfigurationError` on a malformed file so the caller gets a
    clear, attributed error rather than a raw ``yaml.YAMLError``.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML config at {path}",
            code="CFG_YAML_PARSE",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    return data or {}


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------
class SettingsManager:
    """Owns loading, caching and validation of :class:`ZoyaSettings`.

    Parameters
    ----------
    config_path:
        Override for the YAML file location. Defaults to the value declared in
        :class:`AppConfig` (``config/settings.yaml``).
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path_override: Path | None = (
            Path(config_path).resolve() if config_path else None
        )
        self._settings: ZoyaSettings | None = None

    # ------------------------------------------------------------------ load
    def load(self, *, force: bool = False) -> ZoyaSettings:
        """Build and cache the global :class:`ZoyaSettings`.

        Subsequent calls return the cached object unless ``force=True``
        (which is what :meth:`reload` does).
        """
        if self._settings is not None and not force:
            return self._settings

        app = AppConfig()

        yaml_path = self._resolve_config_path(app)
        raw = _load_yaml(yaml_path)

        try:
            automation = AutomationSettings(**raw.get("automation", {}))
            paths = PathSettings(**raw.get("paths", {}))
            llm = LLMSettings(**raw.get("llm", {}))
        except Exception as exc:  # pydantic.ValidationError or similar
            raise ConfigurationError(
                "Invalid values in settings.yaml",
                code="CFG_YAML_INVALID",
                context={"path": str(yaml_path)},
                cause=exc,
            ) from exc

        self._settings = ZoyaSettings(
            app=app, automation=automation, paths=paths, llm=llm
        )
        return self._settings

    def reload(self) -> ZoyaSettings:
        """Discard the cache and re-read everything from disk."""
        self._settings = None
        return self.load(force=True)

    @property
    def settings(self) -> ZoyaSettings:
        """Cached settings; loads lazily on first access."""
        if self._settings is None:
            return self.load()
        return self._settings

    # ------------------------------------------------------------------ check
    def check_health(self, settings: ZoyaSettings | None = None) -> HealthReport:
        """Run environment-aware sanity checks.

        Rules
        -----
        * ``production`` **without** an API key   → *error*
        * ``development``/``staging`` without key → *warning*
        * automation disabled at runtime          → *warning* (easy to forget)
        """
        s = settings or self.settings
        report = HealthReport()

        if not s.app.has_api_key:
            if s.is_production:
                report.add_error(
                    "GEMINI_API_KEY is required in production but is not set."
                )
            else:
                report.add_warning(
                    "GEMINI_API_KEY is not set; the LLM brain will be disabled."
                )

        if not s.app.automation_enabled:
            report.add_warning(
                "AUTOMATION_ENABLED is false — desktop control is disabled."
            )

        if s.app.automation_dry_run and not s.is_development:
            report.add_warning(
                "AUTOMATION_DRY_RUN is on outside development — "
                "no real input will be sent."
            )

        return report

    # ------------------------------------------------------------------ utils
    def _resolve_config_path(self, app: AppConfig) -> Path:
        if self._config_path_override is not None:
            return self._config_path_override
        return (PATHS.base / app.config_path).resolve()


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------
_default_manager: SettingsManager | None = None


def get_settings_manager() -> SettingsManager:
    """Return the process-wide :class:`SettingsManager` (created on demand)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SettingsManager()
    return _default_manager


def load_settings() -> ZoyaSettings:
    """Convenience accessor — equivalent to ``get_settings_manager().settings``.

    Kept for backwards-compatibility with the rest of the codebase
    (``from zoya.core.config import load_settings``).
    """
    return get_settings_manager().settings


def reset_settings_manager() -> None:
    """Drop the cached singleton + loaded settings.

    Primarily for tests that need a pristine configuration between cases.
    """
    global _default_manager
    _default_manager = None


__all__ = [
    "HealthReport",
    "SettingsManager",
    "get_settings_manager",
    "load_settings",
    "reset_settings_manager",
]
