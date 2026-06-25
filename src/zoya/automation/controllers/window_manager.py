"""Window management via the Win32 API (:mod:`pywin32`).

Responsibility (SRP): enumerate and manipulate *top-level windows* — list,
find, focus, minimise/maximise/restore/close and move/resize.

Windows-only. The ``win32*`` imports are guarded so importing this module on
another OS does not crash the whole package; instead, instantiating
:class:`WindowManager` off-Windows raises a clear error.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

from zoya.core.exceptions import AutomationError, WindowNotFoundError
from zoya.core.logging import get_logger

logger = get_logger("automation.window")

_IS_WINDOWS = sys.platform == "win32"

try:  # Windows-only dependency; guard so the package imports anywhere.
    import win32con  # type: ignore
    import win32gui  # type: ignore
except ImportError:  # pragma: no cover - only exercised off Windows
    win32gui = None  # type: ignore
    win32con = None  # type: ignore


@dataclass(frozen=True)
class WindowInfo:
    """Immutable description of a window at a point in time."""

    hwnd: int
    title: str
    class_name: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom)

    @property
    def size(self) -> tuple[int, int]:
        left, top, right, bottom = self.rect
        return (right - left, bottom - top)


def _ensure_windows() -> None:
    """Raise a descriptive error if Win32 APIs are unavailable."""
    if not _IS_WINDOWS or win32gui is None:
        raise AutomationError(
            "Window management is only available on Windows with pywin32 installed."
        )


class WindowManager:
    """Wrapper around the subset of Win32 calls Zoya needs."""

    def __init__(self) -> None:
        _ensure_windows()

    # ------------------------------------------------------------------ #
    # Enumeration / lookup                                                #
    # ------------------------------------------------------------------ #
    def list_windows(self, include_empty: bool = False) -> List[WindowInfo]:
        """Return all visible top-level windows, sorted by title.

        ``include_empty`` keeps windows that have no title (usually background
        / helper windows) which are filtered out by default for readability.
        """
        results: List[WindowInfo] = []

        def _enum(hwnd: int, _ctx: object) -> bool:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title or include_empty:
                    results.append(
                        WindowInfo(
                            hwnd=hwnd,
                            title=title,
                            class_name=win32gui.GetClassName(hwnd),
                            rect=win32gui.GetWindowRect(hwnd),
                        )
                    )
            return True  # continue enumeration

        win32gui.EnumWindows(_enum, None)
        results.sort(key=lambda w: w.title.lower())
        return results

    def find_window(self, title: str, exact: bool = False) -> Optional[int]:
        """Return the window handle whose title matches.

        Matching is case-insensitive substring by default; pass
        ``exact=True`` for an exact title comparison. Returns ``None`` when no
        window matches.
        """
        needle = title.strip().lower()
        if not needle:
            return None
        for w in self.list_windows():
            hay = w.title.lower()
            if (hay == needle) if exact else (needle in hay):
                return w.hwnd
        return None

    def get_active(self) -> Optional[WindowInfo]:
        """Describe the window currently in the foreground."""
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        return WindowInfo(
            hwnd=hwnd,
            title=win32gui.GetWindowText(hwnd),
            class_name=win32gui.GetClassName(hwnd),
            rect=win32gui.GetWindowRect(hwnd),
        )

    # ------------------------------------------------------------------ #
    # Mutations                                                           #
    # ------------------------------------------------------------------ #
    def _require(self, title: str) -> int:
        hwnd = self.find_window(title)
        if not hwnd:
            raise WindowNotFoundError(f"No window matching {title!r}")
        return hwnd

    def focus(self, hwnd: int) -> None:
        """Bring a window to the foreground.

        ``SetForegroundWindow`` is restricted by Windows unless our process
        already owns the foreground. The classic workaround — synthesise an
        inert ALT keypress first — relaxes that restriction, so we use it.
        """
        # Restore first in case the window is minimised.
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        # ALT-down / ALT-up trick to obtain foreground permission.
        win32gui.keybd_event(0x12, 0, 0, 0)
        win32gui.keybd_event(0x12, 0, 0x0002, 0)
        win32gui.SetForegroundWindow(hwnd)

    def minimize(self, hwnd: int) -> None:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

    def maximize(self, hwnd: int) -> None:
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

    def restore(self, hwnd: int) -> None:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    def close(self, hwnd: int) -> None:
        """Close a window gracefully by posting ``WM_CLOSE`` (lets the app save
        / prompt)."""
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    def move_resize(self, hwnd: int, x: int, y: int, width: int, height: int) -> None:
        """Reposition and resize a window in one call."""
        win32gui.MoveWindow(hwnd, int(x), int(y), int(width), int(height), True)

    # Convenience: perform an action resolved by title string directly.
    def apply(self, title: str, action: Callable[[int], None]) -> int:
        """Find a window by title, apply ``action(hwnd)``, return the handle."""
        hwnd = self._require(title)
        action(hwnd)
        return hwnd


__all__ = ["WindowInfo", "WindowManager"]
