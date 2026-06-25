"""Application bootstrap & lifecycle for Zoya.

The bootstrap is the *single seam* between "the interpreter just started" and
"Zoya is ready to serve". It wires the foundation together in the correct
order:

    1. load + validate configuration (``SettingsManager``)
    2. configure logging from that configuration (``setup_logging``)
    3. register the API key for redaction so it never leaks into logs
    4. ensure runtime directories exist
    5. run health checks; abort in production if there are hard errors

Everything that needs an early-failing, well-logged startup should call
:func:`bootstrap` once (idempotent) and keep the returned
:class:`RuntimeContext`. :func:`shutdown` performs the symmetric teardown.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from zoya.core.config.manager import (
    HealthReport,
    SettingsManager,
    get_settings_manager,
    reset_settings_manager,
)
from zoya.core.config.models import ZoyaSettings
from zoya.core.exceptions import BootstrapError
from zoya.core.logging import (
    get_logger,
    register_secret,
    setup_logging,
    shutdown_logging,
)
from zoya.core.paths import PATHS, ProjectPaths

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeContext:
    """The bundle of initialised services handed back by :func:`bootstrap`.

    Treat this as the single dependency a top-level entry point needs; pass
    individual pieces (``settings``, ``paths``, ``logger``) down to subsystems.
    """

    settings: ZoyaSettings
    paths: ProjectPaths
    logger: logging.Logger
    health: HealthReport


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap(
    *,
    settings_manager: SettingsManager | None = None,
    ensure_dirs: Sequence[str] = ("logs", "screenshots", "notes"),
    fail_on_health_errors: bool | None = None,
    json_logs: bool | None = None,
) -> RuntimeContext:
    """Initialise the foundation and return a :class:`RuntimeContext`.

    Idempotent in spirit: a second call re-runs the steps (useful after a
    config edit) but never duplicates logging handlers.

    Parameters
    ----------
    settings_manager:
        Inject a custom manager (mainly for tests). Defaults to the shared
        singleton, which is reset first so a stale cache never survives.
    ensure_dirs:
        Runtime directories to create. Pass an empty sequence to skip.
    fail_on_health_errors:
        Whether a hard health error raises :class:`BootstrapError`. Defaults to
        ``True`` in production, ``False`` elsewhere.
    json_logs:
        Force JSON / human logs. ``None`` reads ``ZOYA_JSON_LOGS`` from the
        environment so it can be toggled without code changes.
    """
    reset_settings_manager()
    manager = settings_manager or get_settings_manager()

    # 1) Load configuration (raises ConfigurationError on malformed input).
    try:
        settings = manager.load(force=True)
    except Exception as exc:
        # Logging may not be up yet; print to stderr as a last resort.
        print(f"[bootstrap] fatal configuration error: {exc}", file=sys.stderr)
        raise

    # 2) Configure logging from the freshly-loaded settings.
    use_json = json_logs
    if use_json is None:
        use_json = _env_bool("ZOYA_JSON_LOGS", default=False)
    logger = setup_logging(
        level=settings.app.log_level,
        log_dir=settings.log_path,
        json_logs=use_json,
    )
    boot_logger = get_logger("core.bootstrap")
    boot_logger.info(
        "Bootstrapping Zoya | env=%s | model=%s | dry_run=%s",
        settings.app.environment.value,
        settings.app.gemini_model,
        settings.app.automation_dry_run,
    )

    # 3) Redact the API key (and any obvious secret) from every future record.
    if settings.app.has_api_key:
        register_secret(settings.app.gemini_api_key)

    # 4) Ensure runtime directories.
    if ensure_dirs:
        created = PATHS.ensure_dirs(*ensure_dirs)
        boot_logger.debug("Ensured runtime directories: %s", [str(p) for p in created])

    # 5) Health check.
    health = manager.check_health(settings)
    for warning in health.warnings:
        boot_logger.warning("config: %s", warning)
    for error in health.errors:
        boot_logger.error("config: %s", error)

    strict = settings.is_production if fail_on_health_errors is None else fail_on_health_errors
    if strict and not health.ok:
        raise BootstrapError(
            "Configuration failed production health checks",
            code="BOOT_HEALTH",
            context={"errors": health.errors},
        )

    return RuntimeContext(
        settings=settings,
        paths=PATHS,
        logger=logger,
        health=health,
    )


def shutdown(ctx: RuntimeContext | None = None) -> None:
    """Tear down the foundation (flush logs, drop caches).

    Safe to call even if :func:`bootstrap` was never invoked.
    """
    logger = get_logger("core.bootstrap")
    logger.info("Shutting down Zoya foundation.")
    shutdown_logging()
    reset_settings_manager()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _env_bool(name: str, *, default: bool) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["RuntimeContext", "bootstrap", "shutdown"]
