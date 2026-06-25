"""System-level controls: audio volume, brightness, clipboard, power state.

Responsibility (SRP): talk to the OS-level "system" controls only. All Win32 /
COM dependencies are guarded with try/except so the package imports anywhere;
the methods raise :class:`SystemControlError` (with a clear message) when the
underlying capability is unavailable on the host.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
from typing import Optional

from zoya.core.exceptions import AutomationError, SystemControlError
from zoya.core.logging import get_logger

logger = get_logger("automation.system")

_IS_WINDOWS = sys.platform == "win32"

# --- Windows Core Audio (volume) via pycaw + comtypes ----------------------
try:
    from ctypes import POINTER, cast  # type: ignore

    from comtypes import CLSCTX_ALL  # type: ignore
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore

    _AUDIO_AVAILABLE = True
except Exception:  # pragma: no cover - only when deps are missing
    _AUDIO_AVAILABLE = False

# --- Brightness ------------------------------------------------------------
try:
    import screen_brightness_control as sbc  # type: ignore

    _BRIGHTNESS_AVAILABLE = True
except Exception:  # pragma: no cover
    _BRIGHTNESS_AVAILABLE = False

# --- Clipboard (Win32) -----------------------------------------------------
try:
    import win32clipboard  # type: ignore

    _CLIPBOARD_AVAILABLE = True
except Exception:  # pragma: no cover
    _CLIPBOARD_AVAILABLE = False


def _ensure_windows() -> None:
    if not _IS_WINDOWS:
        raise AutomationError("System controls are only available on Windows.")


class SystemController:
    """High-level access to OS system controls."""

    def __init__(self) -> None:
        # Acquire the master-volume endpoint once; None means unavailable.
        self._vol = self._open_volume_endpoint()
        self._sbc = sbc if _BRIGHTNESS_AVAILABLE else None

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _open_volume_endpoint() -> Optional[object]:
        if not _AUDIO_AVAILABLE:
            return None
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - depends on host audio
            logger.warning("Audio control unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Volume                                                              #
    # ------------------------------------------------------------------ #
    def get_volume(self) -> Optional[int]:
        """Return current master volume as 0..100, or ``None`` if unavailable."""
        if self._vol is None:
            return None
        return round(self._vol.GetMasterVolumeLevelScalar() * 100)

    def set_volume(self, level: int) -> int:
        if self._vol is None:
            raise SystemControlError("Audio control is not available on this system.")
        level = max(0, min(100, int(level)))
        self._vol.SetMasterVolumeLevelScalar(level / 100.0, None)
        logger.info("Volume set to %d%%", level)
        return level

    def mute(self, muted: bool = True) -> bool:
        if self._vol is None:
            raise SystemControlError("Audio control is not available on this system.")
        self._vol.SetMute(1 if muted else 0, None)
        return muted

    def toggle_mute(self) -> bool:
        if self._vol is None:
            raise SystemControlError("Audio control is not available on this system.")
        return self.mute(not bool(self._vol.GetMute()))

    # ------------------------------------------------------------------ #
    # Brightness                                                          #
    # ------------------------------------------------------------------ #
    def get_brightness(self) -> Optional[int]:
        if self._sbc is None:
            return None
        try:
            return int(self._sbc.get_brightness()[0])
        except Exception as exc:  # external monitors / no backlight
            logger.warning("Brightness read failed: %s", exc)
            return None

    def set_brightness(self, level: int) -> int:
        if self._sbc is None:
            raise SystemControlError("Brightness control is not available on this system.")
        level = max(0, min(100, int(level)))
        self._sbc.set_brightness(level)
        logger.info("Brightness set to %d%%", level)
        return level

    # ------------------------------------------------------------------ #
    # Clipboard                                                           #
    # ------------------------------------------------------------------ #
    def get_clipboard(self) -> str:
        _ensure_windows()
        if not _CLIPBOARD_AVAILABLE:
            raise SystemControlError("Clipboard access is unavailable.")
        try:
            win32clipboard.OpenClipboard()
            try:
                # CF_UNICODETEXT covers the common "copy text" case.
                if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                    data = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                else:
                    data = ""
            finally:
                win32clipboard.CloseClipboard()
            return data or ""
        except Exception as exc:
            raise SystemControlError(f"Failed to read clipboard: {exc}") from exc

    def set_clipboard(self, text: str) -> None:
        _ensure_windows()
        if not _CLIPBOARD_AVAILABLE:
            raise SystemControlError("Clipboard access is unavailable.")
        try:
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
            finally:
                win32clipboard.CloseClipboard()
        except Exception as exc:
            raise SystemControlError(f"Failed to write clipboard: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Power                                                               #
    # ------------------------------------------------------------------ #
    def power(self, action: str, confirm: bool = True) -> str:
        """Perform a power action.

        Non-destructive actions (``lock``) run immediately. Destructive ones
        (``sleep`` / ``shutdown`` / ``restart`` / ``logoff``) require
        ``confirm=True``; otherwise :class:`SystemControlError` is raised so
        callers can re-confirm deliberately.
        """
        _ensure_windows()
        action = action.lower()
        destructive = {"sleep", "shutdown", "restart", "logoff"}

        if action == "lock":
            ctypes.windll.user32.LockWorkStation()
            return "locked"

        if action not in destructive:
            raise SystemControlError(f"Unknown power action: {action!r}")

        logger.warning("Power action requested: %s (confirm=%s)", action, confirm)
        if not confirm:
            raise SystemControlError(f"Power action {action!r} requires confirmation.")

        if action == "sleep":
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=False)
        elif action == "shutdown":
            subprocess.run(["shutdown", "/s", "/t", "0"], check=False)
        elif action == "restart":
            subprocess.run(["shutdown", "/r", "/t", "0"], check=False)
        elif action == "logoff":
            subprocess.run(["shutdown", "/l"], check=False)
        return action


__all__ = ["SystemController"]
