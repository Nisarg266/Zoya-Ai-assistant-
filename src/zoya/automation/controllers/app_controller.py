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
   tool useful even before the registry is populated. Each fallback launch is
   logged at WARNING so it is auditable.

Thread safety
-------------
A single ``AppController`` is shared by every ``open_app`` call, and the tool
layer runs each ``_run`` in a worker thread (``asyncio.to_thread``). All access
to the registry dicts is therefore guarded by an :class:`~threading.RLock`.
File parsing happens *outside* the lock (slow I/O); only the final swap of the
resolved dicts is performed under it, and process spawning happens *outside* the
lock so concurrent launches never block each other.

Errors
------
Two **custom exceptions** are defined here, both subclasses of the central
:class:`~zoya.core.exceptions.ProcessError` (and therefore of
:class:`~zoya.core.exceptions.ZoyaError`) so they integrate with the tool
layer's uniform ``except ZoyaError`` handling:

* :class:`ApplicationNotFoundError` — the name is neither registered nor on PATH.
* :class:`AppLaunchError`           — found, but spawning/elevating failed.

A malformed registry raises :class:`~zoya.core.exceptions.ConfigurationError`.

The controller is Windows-aware but import-safe everywhere: Windows-only
behaviour (UAC elevation) is guarded by platform checks.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zoya.core.exceptions import ConfigurationError, ProcessError
from zoya.core.logging import get_logger
from zoya.core.paths import PATHS

logger = get_logger("automation.app")

#: Default location of the application registry, anchored at the project root.
DEFAULT_REGISTRY_PATH: Path = PATHS.base / "config" / "applications.yaml"

#: ShellExecuteW success threshold — a return value strictly greater than this
#: means success; any value <= 32 is an error code (see :data:`_SE_ERRORS`).
_SE_SUCCESS_THRESHOLD = 32

#: Human-readable mapping of ShellExecuteW error codes (subset).
_SE_ERRORS = {
    0: "out of memory",
    2: "file not found",
    3: "path not found",
    5: "access denied",
    8: "not enough memory",
    26: "sharing violation",
    27: "association incomplete",
    28: "DDE timeout",
    29: "DDE failed",
    30: "DDE busy",
    31: "no association",
    32: "DLL not found",
}


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

    Arguments are a ``list[str]`` (never a space-joined string) so each token is
    passed to the child verbatim — paths and arguments containing spaces survive
    intact (no ``str.split`` round-trip).
    """

    model_config = ConfigDict(extra="forbid")

    executable: str = Field(
        ..., min_length=1, description="Bare exe name (resolved via PATH) or absolute path."
    )
    aliases: list[str] = Field(
        default_factory=list, description="Extra friendly names resolving to this app."
    )
    args: list[str] = Field(
        default_factory=list,
        description="Default argv[1:] tokens passed to the executable.",
    )
    working_dir: Optional[str] = Field(None, description="Working directory for the new process.")
    elevated: bool = Field(False, description="If True, request elevation (UAC) on Windows.")


# ---------------------------------------------------------------------------
# Launch result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessLaunchResult:
    """Structured outcome of a successful :meth:`AppController.open_app` call.

    ``args`` is the *effective* ``argv[1:]`` actually passed to the child (the
    registry defaults plus any caller-supplied extras), not a re-joined string.
    """

    requested: str
    canonical: Optional[str]
    executable: str
    pid: Optional[int]
    args: list[str]
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
    verify_on_load:
        When ``True``, log a warning for every registered executable that cannot
        be found on this machine. Never raises — purely diagnostic.
    """

    def __init__(
        self,
        registry_path: Optional[str | Path] = None,
        verify_on_load: bool = False,
    ) -> None:
        self._registry_path: Path = (
            Path(registry_path).resolve() if registry_path else DEFAULT_REGISTRY_PATH
        )
        self._verify_on_load: bool = verify_on_load
        # Guard all access to the two dicts below — see "Thread safety" above.
        self._lock: threading.RLock = threading.RLock()
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
        file or an invalid entry. Parsing happens outside the lock; only the
        final swap of the resolved dicts is performed under it, so concurrent
        readers are never blocked by file I/O and always observe a consistent
        ``(entries, aliases)`` pair.
        """
        entries, aliases = self._parse_registry()
        with self._lock:
            self._entries, self._aliases = entries, aliases
        logger.info(
            "Application registry loaded",
            extra={"count": len(entries), "registry": str(self._registry_path)},
        )
        if self._verify_on_load:
            missing = self.verify()
            if missing:
                logger.warning(
                    "Registered executables not found on this machine",
                    extra={"missing": sorted(missing)},
                )

    def _parse_registry(self) -> tuple[dict[str, ApplicationEntry], dict[str, str]]:
        """Read & validate the YAML, returning fresh ``(entries, aliases)`` dicts.

        Never mutates instance state, so it is safe to run outside the lock.
        """
        path = self._registry_path
        if not path.exists():
            logger.warning(
                "Application registry not found; starting empty (PATH fallback still available)",
                extra={"registry": str(path)},
            )
            return {}, {}

        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                "Failed to parse application registry",
                code="CFG_APP_REGISTRY_PARSE",
                context={"path": str(path)},
                cause=exc,
            ) from exc

        if not isinstance(raw, dict) or not isinstance(raw.get("applications"), dict):
            logger.warning(
                "Registry is missing a top-level 'applications:' mapping; ignoring",
                extra={"registry": str(path)},
            )
            return {}, {}

        entries: dict[str, ApplicationEntry] = {}
        aliases: dict[str, str] = {}
        for canonical, body in (raw["applications"] or {}).items():
            try:
                entry = ApplicationEntry(**(body or {}))
            except ValidationError as exc:
                raise ConfigurationError(
                    f"Invalid entry for application {canonical!r}",
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
                        "Alias clash ignored; keeping first mapping",
                        extra={"alias": a, "requested": key, "kept": aliases[a]},
                    )
                    continue
                aliases.setdefault(a, key)

        return entries, aliases

    # ------------------------------------------------------------------ query
    def list_applications(self) -> list[str]:
        """Canonical friendly names known to the registry (sorted)."""
        with self._lock:
            return sorted(self._entries.keys())

    def resolve(self, name: str) -> Optional[tuple[str, ApplicationEntry]]:
        """Resolve ``name`` to its ``(canonical, entry)`` if registered, else ``None``.

        Case-insensitive; matches canonical names and any declared alias.
        """
        key = name.strip().lower()
        with self._lock:
            canonical = self._aliases.get(key)
            if canonical is None:
                return None
            return canonical, self._entries[canonical]

    def verify(self) -> list[str]:
        """Return canonical names whose executable cannot be located on this machine."""
        with self._lock:
            items = list(self._entries.items())
        return [canonical for canonical, entry in items if not self._locate(entry.executable)]

    # ------------------------------------------------------------------ launch
    def open_app(
        self,
        name: str,
        args: Optional[list[str]] = None,
        working_dir: Optional[str] = None,
    ) -> ProcessLaunchResult:
        """Open an application by friendly name.

        Resolution: registry first (canonical name or alias), then a PATH
        fallback. For a registry hit, ``args`` is *appended* to the entry's
        default arguments (both are ``list[str]``) and ``working_dir`` overrides
        the entry's default.

        Raises
        ------
        ApplicationNotFoundError
            The name is neither registered nor on PATH.
        AppLaunchError
            The app was found but could not be spawned (permissions, format,
            or a declined UAC prompt).
        """
        extra_args = list(args or [])

        # Resolve under the lock, then copy out the scalars we need so the slow
        # spawn happens *outside* the lock (concurrent launches don't block).
        with self._lock:
            resolved = self.resolve(name)
            if resolved is not None:
                canonical, entry = resolved
                executable = entry.executable
                merged_args = list(entry.args) + extra_args
                final_cwd = working_dir or entry.working_dir
                elevated = entry.elevated
                via_registry = True
            else:
                canonical = None
                executable = name
                merged_args = extra_args
                final_cwd = working_dir
                elevated = False
                via_registry = False

        resolved_exe = self._locate(executable)
        if resolved_exe is None:
            raise ApplicationNotFoundError(
                f"Application not found: {name!r} (resolved to {executable!r}). "
                f"Registered apps: {', '.join(self.list_applications()) or '(none)'}",
                context={"requested": name, "executable": executable},
            )

        if not via_registry:
            logger.warning(
                "Application resolved via PATH fallback (not in registry)",
                extra={"requested": name, "executable": resolved_exe},
            )

        if elevated:
            pid = self._launch_elevated(resolved_exe, merged_args, final_cwd, name)
        else:
            pid = self._launch_subprocess(resolved_exe, merged_args, final_cwd, name)

        result = ProcessLaunchResult(
            requested=name,
            canonical=canonical,
            executable=resolved_exe,
            pid=pid,
            args=merged_args,
            elevated=elevated,
            via_registry=via_registry,
        )
        logger.info(
            "Opened application",
            extra={
                "requested": result.requested,
                "canonical": result.canonical,
                "executable": result.executable,
                "pid": result.pid,
                "elevated": result.elevated,
                "via_registry": result.via_registry,
                "args": result.args,
            },
        )
        return result

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
        arg_list: list[str],
        working_dir: Optional[str],
        requested: str,
    ) -> int:
        """Launch via :class:`subprocess.Popen` and return the new PID.

        Raises :class:`AppLaunchError` on any spawning failure.
        """
        cmd = [executable, *arg_list]
        # CREATE_NEW_PROCESS_GROUP isolates the child from Zoya's Ctrl+C / break
        # signals. It does NOT, by itself, detach the child's lifetime. The
        # attribute is only accessed on win32 (short-circuited away elsewhere),
        # so this stays import-safe on POSIX.
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        logger.info("Launching process", extra={"cmd": cmd})
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
                f"Failed to launch {requested!r} ({executable!r}): {exc}",
                context={
                    "executable": executable,
                    "errno": exc.errno,
                    "winerror": getattr(exc, "winerror", None),
                },
            ) from exc
        return proc.pid

    def _launch_elevated(
        self,
        executable: str,
        arg_list: list[str],
        working_dir: Optional[str],
        requested: str,
    ) -> Optional[int]:
        """Launch with UAC elevation (Windows only) via ``ShellExecuteW``.

        Returns ``None`` because ``ShellExecuteW`` does not expose the child PID.

        Raises :class:`AppLaunchError` on non-Windows platforms or a failed /
        cancelled elevation.
        """
        if sys.platform != "win32":
            raise AppLaunchError(
                f"Elevation is only supported on Windows (requested for {requested!r}).",
                context={"executable": executable, "platform": sys.platform},
            )

        # Declare proper signatures: without them ctypes defaults restype to a
        # 32-bit c_int, which TRUNCATES the pointer-sized HINSTANCE on 64-bit
        # Windows and makes the success/failure check unreliable.
        from ctypes import wintypes

        shell_execute = ctypes.windll.shell32.ShellExecuteW
        shell_execute.argtypes = [
            wintypes.HWND,    # hwnd
            wintypes.LPCWSTR, # lpVerb
            wintypes.LPCWSTR, # lpFile
            wintypes.LPCWSTR, # lpParameters
            wintypes.LPCWSTR, # lpDirectory
            wintypes.UINT,    # nShowCmd
        ]
        shell_execute.restype = wintypes.HINSTANCE

        # ShellExecuteW takes lpParameters as ONE string; join argv with proper
        # Windows quoting so paths/args containing spaces survive.
        params = subprocess.list2cmdline(arg_list) if arg_list else None
        SW_SHOWNORMAL = 1

        rc = int(
            shell_execute(None, "runas", executable, params, working_dir, SW_SHOWNORMAL)
        )
        if rc <= _SE_SUCCESS_THRESHOLD:
            reason = _SE_ERRORS.get(rc, "unknown error")
            raise AppLaunchError(
                f"Elevated launch failed for {requested!r}: {reason} "
                f"(ShellExecute code={rc}). The user may have declined the UAC "
                "prompt or the file was not found.",
                context={"executable": executable, "shell_execute_code": rc, "reason": reason},
            )
        logger.warning(
            "Elevated launch started; child PID is not reported by ShellExecuteW",
            extra={"requested": requested, "executable": executable},
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
