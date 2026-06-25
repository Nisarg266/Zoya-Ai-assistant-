"""Process & application management via :mod:`psutil` + :mod:`subprocess`.

Responsibility (SRP): launch, query and terminate *processes / applications*.
Launch resolution tries the system PATH (``shutil.which``) and a small alias
table so callers can say ``"notepad"`` instead of ``"C:\\Windows\\notepad.exe"``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Union

import psutil

from zoya.core.exceptions import ProcessError
from zoya.core.logging import get_logger

logger = get_logger("automation.process")


@dataclass(frozen=True)
class ProcessInfo:
    """Minimal description of a running process."""

    pid: int
    name: str
    username: Optional[str]


# Friendly application aliases -> executable file names. Extend as needed.
_APP_ALIASES = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "taskmgr": "taskmgr.exe",
    "vscode": "code.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
}


class ProcessManager:
    """Launch, find and terminate processes."""

    def __init__(self, launch_timeout: float = 10.0) -> None:
        self._launch_timeout = max(0.0, launch_timeout)

    # ------------------------------------------------------------------ #
    # Launch                                                              #
    # ------------------------------------------------------------------ #
    def launch_application(
        self,
        target: str,
        args: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> subprocess.Popen:
        """Launch an application.

        ``target`` may be a bare name resolved via PATH / alias table, or an
        absolute path to an executable. ``args`` is a single string split on
        whitespace for convenience (use a list-based API if you need quoting).
        """
        exe = shutil.which(target) or self._resolve_alias(target) or target
        cmd = [exe]
        if args:
            cmd += args.split()

        try:
            logger.info("Launching: %s", " ".join(cmd))
            return subprocess.Popen(cmd, cwd=working_dir, shell=False)
        except FileNotFoundError as exc:
            raise ProcessError(f"Application not found: {target}") from exc
        except Exception as exc:
            raise ProcessError(f"Failed to launch {target}: {exc}") from exc

    @staticmethod
    def _resolve_alias(name: str) -> Optional[str]:
        return _APP_ALIASES.get(name.lower())

    # ------------------------------------------------------------------ #
    # Query                                                               #
    # ------------------------------------------------------------------ #
    def list_processes(self, name_filter: Optional[str] = None) -> List[ProcessInfo]:
        """List running processes, optionally filtered by substring of name."""
        needle = name_filter.lower() if name_filter else None
        out: List[ProcessInfo] = []
        for proc in psutil.process_iter(attrs=["pid", "name", "username"]):
            try:
                info = proc.info
                name = info.get("name") or ""
                if needle and needle not in name.lower():
                    continue
                out.append(ProcessInfo(info.get("pid"), name, info.get("username")))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process vanished or is protected — skip silently.
                continue
        return out

    def find_process(self, name: str) -> Optional[ProcessInfo]:
        matches = self.list_processes(name_filter=name)
        return matches[0] if matches else None

    def is_running(self, name: str) -> bool:
        return self.find_process(name) is not None

    # ------------------------------------------------------------------ #
    # Termination                                                         #
    # ------------------------------------------------------------------ #
    def terminate_process(self, target: Union[int, str], force: bool = False) -> bool:
        """Terminate one process (by pid) or all matching a name (substring).

        ``force=True`` uses SIGKILL; otherwise a graceful terminate is issued.
        Raises :class:`ProcessError` if nothing matched.
        """
        targets: List[psutil.Process] = []

        if isinstance(target, int):
            try:
                targets = [psutil.Process(target)]
            except psutil.NoSuchProcess as exc:
                raise ProcessError(f"No process with pid {target}") from exc
        else:
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    pname = proc.info.get("name") or ""
                    if target.lower() in pname.lower():
                        targets.append(psutil.Process(proc.info["pid"]))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        if not targets:
            raise ProcessError(f"No process matching {target!r}")

        for proc in targets:
            try:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
                logger.warning("Terminated %s (pid=%s, force=%s)", proc.name(), proc.pid, force)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                logger.error("Could not terminate pid=%s: %s", proc.pid, exc)
        return True


__all__ = ["ProcessInfo", "ProcessManager"]
