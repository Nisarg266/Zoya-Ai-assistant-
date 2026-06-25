"""Mouse input simulation built on :mod:`pynput`.

Responsibility (SRP): *mouse only* — move, click, drag, scroll. The cursor
movement is interpolated by default so it looks human and avoids the abrupt
teleports that can trigger OS anti-tamper / failsafe logic.
"""

from __future__ import annotations

import logging
import time

from pynput import mouse
from pynput.mouse import Button

from zoya.core.exceptions import InputSimulationError
from zoya.core.logging import get_logger

logger = get_logger("automation.mouse")

# Friendly button names -> pynput Button.
_BUTTON_MAP = {
    "left": Button.left,
    "right": Button.right,
    "middle": Button.middle,
    "centre": Button.middle,  # UK spelling convenience
}


class MouseController:
    """High-level wrapper around :class:`pynput.mouse.Controller`."""

    def __init__(
        self,
        move_duration: float = 0.3,
        move_steps: int = 50,
        click_interval: float = 0.1,
    ) -> None:
        self._mouse = mouse.Controller()
        self._move_duration = max(0.0, move_duration)
        self._move_steps = max(1, move_steps)
        self._click_interval = max(0.0, click_interval)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _button(self, name: str) -> Button:
        try:
            return _BUTTON_MAP[name.lower()]
        except KeyError:
            raise InputSimulationError(f"Unknown mouse button: {name!r}")

    # ------------------------------------------------------------------ #
    # Positioning                                                         #
    # ------------------------------------------------------------------ #
    def position(self) -> tuple[int, int]:
        """Return the current cursor position as ``(x, y)``."""
        return self._mouse.position

    def move(self, x: int, y: int, duration: float | None = None, smooth: bool = True) -> None:
        """Move to absolute ``(x, y)``.

        With ``smooth=True`` (default) the movement is interpolated across
        ``duration`` seconds; otherwise the cursor jumps instantly.
        """
        duration = self._move_duration if duration is None else max(0.0, duration)
        start_x, start_y = self._mouse.position

        if not smooth or duration <= 0:
            self._mouse.position = (int(x), int(y))
            return

        steps = self._move_steps
        sleep = duration / steps
        # Linear interpolation in `steps` chunks.
        for i in range(1, steps + 1):
            t = i / steps
            nx = int(start_x + (x - start_x) * t)
            ny = int(start_y + (y - start_y) * t)
            self._mouse.position = (nx, ny)
            time.sleep(sleep)

    # ------------------------------------------------------------------ #
    # Clicking / dragging                                                 #
    # ------------------------------------------------------------------ #
    def click(self, button: str = "left", clicks: int = 1, interval: float | None = None) -> None:
        btn = self._button(button)
        delay = self._click_interval if interval is None else max(0.0, interval)
        for _ in range(max(1, clicks)):
            self._mouse.click(btn)
            if delay:
                time.sleep(delay)

    def double_click(self, button: str = "left") -> None:
        self.click(button=button, clicks=2)

    def right_click(self) -> None:
        self.click(button="right", clicks=1)

    def press(self, button: str = "left") -> None:
        self._mouse.press(self._button(button))

    def release(self, button: str = "left") -> None:
        self._mouse.release(self._button(button))

    def drag(self, x: int, y: int, duration: float = 0.5, button: str = "left") -> None:
        """Move to ``(x, y)`` while holding ``button`` — a drag operation."""
        self.press(button)
        try:
            self.move(x, y, duration=duration, smooth=True)
        finally:
            # Always release, even if movement raised.
            self.release(button)

    # ------------------------------------------------------------------ #
    # Scrolling                                                           #
    # ------------------------------------------------------------------ #
    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        """Scroll the wheel. Positive ``dy`` scrolls up; positive ``dx`` right."""
        self._mouse.scroll(int(dx), int(dy))


__all__ = ["MouseController"]
