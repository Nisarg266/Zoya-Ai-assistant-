"""Application launcher controller — open installed apps by friendly name.

Responsibility (SRP): resolve a *friendly name* (e.g. ``"notepad"``,
``"calculator"``) to a real executable using a configurable YAML registry, then
launch it with :mod:`subprocess`.

Resolution order
----------------
1. **Registry** (``config/applications.yaml``): canonical names + declared
   ``aliases`` (case-insensitive).
2. **System PATH** fallback: if the name isn't registered, it is treated as a
   bare executable name and resolved with :func:`shutil.which`. This keeps the
   tool useful even before the registry is populated.

Errors
------
This module defines two **custom exceptions**, both subclasses of the existing
:class:`~zoya.core.exceptions.ProcessError` (and therefore of
:class:`~zoya.core.exceptions.ZoyaError`) so they integrate with the tool
layer's uniform ``except ZoyaError`` handling:

* :class:`ApplicationNotFoundError` — the name is neither registered nor on PATH.
* :class:`AppLaunchError`           — found, but spawning/elevating failed.

A malformed registry still raises :class:`~zoya.core.exceptions.ConfigurationError`.

This controller is intentionally Windows-aware but import-safe everywhere:
Windows-only behaviour (UAC elevation) is guarded by platform checks.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zoya.core.exceptions import ConfigurationError, ProcessError
from zoya.core.logging import get_logger
from zoya.core.paths import PATHS

logger = get_logger("automation.app")

#: Default location of the application registry, anchored at the project root.
DEFAULT_REGISTRY_PATH: Path = PATHS.base / "config" / "applications.yaml"


# ---------------------------------------------------------------------------
# Custom exceptions (module-local; subclass the central ProcessError)
# ---------------------------------------------------------------------------
class ApplicationNotFoundError(ProcessError):
    """The requested friendly name is neither registered nor on the system PATH.

    This is a *lookup* failure: Zoya never even tried to spawn a process.
    """

    default_code = "AUTO_APP_NOT_FOUND"


class AppLaunchError(ProcessError):
    """A resolved application could not be spawned (subprocess / UAC failure).

    The executable was located, but :mod:`subprocess` (or ``ShellExecuteW`` for
    elevated launches) failed — e.g. permissions, exec format, or the user
    declined the UAC prompt.
    """

    default_code = "AUTO_APP_LAUNCH"


# ---------------------------------------------------------------------------
# Registry entry model
# ---------------------------------------------------------------------------
class ApplicationEntry(BaseModel):
    """One application as declared in ``config/applications.yaml``.

    ``extra="forbid"`` so a typo in a field name (e.g. ``executabel:``) surfaces
    as a precise :class:`~zoya.core.exceptions.ConfigurationError` at load time
    instead of silently being ignored.
    """

    model_config = ConfigDict(extra="forbid")

    executable: str = Field(
        ..., min_length=1, description="Bare exe name (resolved via PATH) or absolute path."
    )
    aliases: List[str] = Field(
        default_factory=list, description="Extra friendly names resolving to this app."
    )
    args: Optional[str] = Field(None, description="Default command-line arguments.")
    working_dir: Optional[str] = Field(None, description="Working directory for the new process.")
    elevated: bool = Field(False, description="If True, request elevation (UAC) on Windows.")


# ---------------------------------------------------------------------------
# Launch result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessLaunchResult:
    """Structured outcome of a successful :meth:`AppController.open_app` call."""

    name: str
    executable: str
    pid: Optional[int]
    args: str
    elevated: bool
    via_registry: bool


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class AppController:
    """Resolve friendly names via a YAML registry and launch applications.

    Parameters
    ----------
    registry_path:
        Override for the registry file. Defaults to
        :data:`DEFAULT_REGISTRY_PATH` (``config/applications.yaml``).
    launch_timeout:
        Reserved upper bound (seconds) for future wait-on-launch behaviour.
        Kept for parity with
        :class:`~zoya.automation.controllers.process_manager.ProcessManager` and
        driven from ``automation.launch_timeout`` in ``settings.yaml``.
    verify_on_load:
        When ``True``, log a warning for every registered executable that cannot
        be found on this machine. Never raises — purely diagnostic.
    """

    def __init__(
        self,
        registry_path: Optional[str | Path] = None,
        launch_timeout: float = 10.0,
        verify_on_load: bool = False,
    ) -> None:
        self._registry_path: Path = (
            Path(registry_path).resolve() if registry_path else DEFAULT_REGISTRY_PATH
        )
        self._launch_timeout: float = max(0.0, launch_timeout)
        self._verify_on_load: bool = verify_on_load
        self._entries: dict[str, ApplicationEntry] = {}
        self._aliases: dict[str, str] = {}  # alias(lower) -> canonical(lower)
        self.reload()

    # ------------------------------------------------------------------ load
    @property
    def registry_path(self) -> Path:
        """Absolute path of the registry file in use."""
        return self._registry_path

    def reload(self) -> None:
        """(Re)read the registry from disk and rebuild the alias index.

        Raises :class:`~zoya.core.exceptions.ConfigurationError` on a malformed
        file or an invalid entry; otherwise idempotent and safe to call anytime
        (e.g. after the user edits the YAML while Zoya is running).
        """
        path = self._registry_path
        if not path.exists():
            logger.warning(
                "Application registry not found at %s; starting empty "
                "(PATH fallback still available).",
                path,
            )
            self._entries, self._aliases = {}, {}
            return

        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse application registry at {path}",
                code="CFG_APP_REGISTRY_PARSE",
                context={"path": str(path)},
                cause=exc,
            ) from exc

        if not isinstance(raw, dict) or not isinstance(raw.get("applications"), dict):
            logger.warning(
                "Registry %s is missing a top-level 'applications:' mapping; ignoring.",
                path,
            )
            self._entries, self._aliases = {}, {}
            return

        entries: dict[str, ApplicationEntry] = {}
        aliases: dict[str, str] = {}
        for canonical, body in (raw["applications"] or {}).items():
            try:
                entry = ApplicationEntry(**(body or {}))
            except ValidationError as exc:
                raise ConfigurationError(
                    f"Invalid entry for application {canonical!r} in {path}",
                    code="CFG_APP_REGISTRY_INVALID",
                    context={"app": canonical, "path": str(path)},
                    cause=exc,
                ) from exc

            key = str(canonical).strip().lower()
            entries[key] = entry
            aliases.setdefault(key, key)
            for alias in entry.aliases:
                a = str(alias).strip().lower()
                if not a:
                    continue
                if a in aliases and aliases[a] != key:
                    logger.warning(
                        "Alias %r for %r clashes with %r; keeping the first mapping.",
                        a,
                        key,
                        aliases[a],
                    )
                    continue
                aliases.setdefault(a, key)

        self._entries, self._aliases = entries, aliases
        logger.info("Loaded %d application(s) from registry (%s).", len(entries), path)
        if self._verify_on_load:
            missing = self.verify()
            if missing:
                logger.warning(
                    "Registered executables not found on this machine: %s",
                    ", ".join(sorted(missing)),
                )

    # ------------------------------------------------------------------ query
    def list_applications(self) -> List[str]:
        """Canonical friendly names known to the registry (sorted)."""
        return sorted(self._entries.keys())

    def resolve(self, name: str) -> Optional[tuple[str, ApplicationEntry]]:
        """Resolve ``name`` to its ``(canonical, entry)`` if registered, else ``None``.

        Lookup is case-insensitive and matches canonical names and any declared
        alias.
        """
        key = name.strip().lower()
        canonical = self._aliases.get(key)
        if canonical is None:
            return None
        return canonical, self._entries[canonical]

    def verify(self) -> List[str]:
        """Return canonical names whose executable cannot be located on this machine."""
        missing: List[str] = []
        for canonical, entry in self._entries.items():
            if not self._locate(entry.executable):
                missing.append(canonical)
        return missing

    # ------------------------------------------------------------------ launch
    def open_app(
        self,
        name: str,
        args: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> ProcessLaunchResult:
        """Open an application by friendly name.

        Resolution: registry first (canonical name or alias), then a PATH
        fallback so bare exe names still work. For a registry hit, ``args`` is
        *appended* to the entry's default arguments and ``working_dir`` overrides
        the entry's default.

        Raises
        ------
        ApplicationNotFoundError
            The name is neither registered nor on PATH.
        AppLaunchError
            The app was found but could not be spawned (permissions, format,
            or a declined UAC prompt).
        """
        resolved = self.resolve(name)
        if resolved is not None:
            canonical, entry = resolved
            executable = entry.executable
            merged_args = " ".join(s for s in (entry.args, args) if s)
            final_cwd = working_dir or entry.working_dir
            elevated = entry.elevated
            via_registry = True
            display_name = canonical
        else:
            executable = name
            merged_args = args or ""
            final_cwd = working_dir
            elevated = False
            via_registry = False
            display_name = name

        resolved_exe = self._locate(executable)
        if resolved_exe is None:
            raise ApplicationNotFoundError(
                f"Application not found: {name!r} (resolved to {executable!r}). "
                f"Registered apps: {', '.join(self.list_applications()) or '(none)'}",
                context={"requested": name, "executable": executable},
            )

        arg_list = merged_args.split() if merged_args else []

        if elevated:
            pid = self._launch_elevated(resolved_exe, arg_list, final_cwd, display_name)
        else:
            pid = self._launch_subprocess(resolved_exe, arg_list, final_cwd, display_name, name)

        logger.info(
            "Opened %r -> %s (pid=%s, elevated=%s, via_registry=%s).",
            display_name,
            resolved_exe,
            pid if pid is not None else "n/a",
            elevated,
            via_registry,
        )
        return ProcessLaunchResult(
            name=display_name,
            executable=resolved_exe,
            pid=pid,
            args=merged_args,
            elevated=elevated,
            via_registry=via_registry,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _locate(executable: str) -> Optional[str]:
        """Resolve ``executable`` to an absolute path via PATH or direct existence.

        Returns ``None`` if it can't be found anywhere.
        """
        return shutil.which(executable) or (
            str(path) if (path := Path(executable)).is_file() else None
        )

    def _launch_subprocess(
        self,
        executable: str,
        arg_list: List[str],
        working_dir: Optional[str],
        display_name: str,
        original_name: str,
    ) -> int:
        """Launch via :class:`subprocess.Popen` and return the new PID.

        Raises :class:`AppLaunchError` on any spawning failure.
        """
        cmd = [executable, *arg_list]
        # Detach the child into its own process group on Windows so it survives
        # Zoya. The attribute is only accessed on win32 (short-circuited away
        # elsewhere), so this stays import-safe on POSIX.
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        logger.info("Launching: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=working_dir,
                shell=False,
                close_fds=True,
                creationflags=creationflags,
            )
        except OSError as exc:  # FileNotFoundError / PermissionError / etc.
            raise AppLaunchError(
                f"Failed to launch {original_name!r} ({executable!r}): {exc}",
                context={"executable": executable, "errno": getattr(exc, "winerror", None)},
            ) from exc
        return proc.pid

    def _launch_elevated(
        self,
        executable: str,
        arg_list: List[str],
        working_dir: Optional[str],
        display_name: str,
    ) -> Optional[int]:
        """Launch with UAC elevation (Windows only) via ``ShellExecuteW``.

        Returns ``None`` because ``ShellExecuteW`` does not expose the child PID.

        Raises :class:`AppLaunchError` on non-Windows platforms or a failed /
        cancelled elevation.
        """
        if sys.platform != "win32":
            raise AppLaunchError(
                f"Elevation is only supported on Windows (requested for {display_name!r}).",
                context={"executable": executable, "platform": sys.platform},
            )

        import ctypes

        SW_SHOWNORMAL = 1
        params = " ".join(arg_list) if arg_list else None
        # ShellExecuteW(hwnd, verb, file, params, directory, showCmd) -> int > 32 on success.
        rc: Any = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", executable, params, working_dir, SW_SHOWNORMAL
        )
        rc = int(rc)
        if rc <= 32:
            # 2 = FILE_NOT_FOUND, 5 = ACCESS_DENIED, 1223 = user cancelled UAC.
            raise AppLaunchError(
                f"Elevated launch failed for {display_name!r} "
                f"(ShellExecute code={rc}). The user may have declined the UAC "
                "prompt or the file was not found.",
                context={"executable": executable, "shell_execute_code": rc},
            )
        logger.warning(
            "Elevated launch started for %r; child PID is not reported by ShellExecuteW.",
            display_name,
        )
        return None


__all__ = [
    "AppController",
    "ApplicationEntry",
    "ProcessLaunchResult",
    "ApplicationNotFoundError",
    "AppLaunchError",
    "DEFAULT_REGISTRY_PATH",
]
