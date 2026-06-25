"""Keyboard input simulation built on :mod:`pynput`.

Responsibility (SRP): *keyboard only* — type text, tap keys, emit hotkeys.
No mouse, no windows, no business logic. Higher layers (tools) orchestrate.

A key design choice is the **alias table** below: callers (the LLM included)
describe keys with friendly strings like ``"ctrl"``, ``"win"`` or ``"f5"``.
``_resolve_key`` translates those into either a pynput :class:`Key` member or a
single character, raising :class:`InputSimulationError` on anything unknown so
typos surface immediately instead of silently doing nothing.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Union

from pynput import keyboard
from pynput.keyboard import Key

from zoya.core.exceptions import InputSimulationError
from zoya.core.logging import get_logger

logger = get_logger("automation.keyboard")

# A resolved key is either a special Key enum member or a single character.
ResolvedKey = Union[Key, str]

# Human-friendly aliases -> pynput Key. Everything is matched case-insensitively
# after lower-casing the token.
_KEY_ALIASES: Dict[str, Key] = {
    # Modifiers
    "ctrl": Key.ctrl, "control": Key.ctrl, "ctrl_l": Key.ctrl_l, "ctrl_r": Key.ctrl_r,
    "alt": Key.alt, "alt_l": Key.alt_l, "alt_r": Key.alt_r, "option": Key.alt,
    "shift": Key.shift, "shift_l": Key.shift_l, "shift_r": Key.shift_r,
    "cmd": Key.cmd, "cmd_l": Key.cmd_l, "cmd_r": Key.cmd_r,
    "win": Key.cmd, "super": Key.cmd, "meta": Key.cmd,
    # Editing / navigation
    "enter": Key.enter, "return": Key.enter,
    "esc": Key.esc, "escape": Key.esc,
    "tab": Key.tab, "space": Key.space, "backspace": Key.backspace,
    "delete": Key.delete, "del": Key.delete,
    "insert": Key.insert,
    "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
    "home": Key.home, "end": Key.end,
    "page_up": Key.page_up, "pageup": Key.page_up, "pgup": Key.page_up,
    "page_down": Key.page_down, "pagedown": Key.page_down, "pgdn": Key.page_down,
    # Lock / special keys
    "caps_lock": Key.caps_lock, "capslock": Key.caps_lock,
    "num_lock": Key.num_lock, "numlock": Key.num_lock,
    "scroll_lock": Key.scroll_lock,
    "print_screen": Key.print_screen, "prtsc": Key.print_screen, "printscreen": Key.print_screen,
    "menu": Key.menu, "pause": Key.pause,
}
# Function keys F1..F24 (only the ones pynput actually defines).
for _i in range(1, 25):
    _fname = f"f{_i}"
    if hasattr(Key, _fname):
        _KEY_ALIASES[_fname] = getattr(Key, _fname)


class KeyboardController:
    """High-level wrapper around :class:`pynput.keyboard.Controller`."""

    def __init__(self, type_interval: float = 0.0, key_interval: float = 0.1) -> None:
        self._kb = keyboard.Controller()
        self._type_interval = max(0.0, type_interval)
        self._key_interval = max(0.0, key_interval)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _resolve_key(self, token: str) -> ResolvedKey:
        """Translate one key token (e.g. ``"ctrl"``, ``"f5"``, ``"a"``) into a
        pynput :class:`Key` or a single character.

        Raises :class:`InputSimulationError` for empty or unrecognised tokens.
        """
        token = token.strip()
        if not token:
            raise InputSimulationError("Empty key token")
        low = token.lower()
        if low in _KEY_ALIASES:
            return _KEY_ALIASES[low]
        # Any single printable character is treated as a literal key.
        if len(token) == 1:
            return token
        raise InputSimulationError(f"Unknown key token: {token!r}")

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def type_text(self, text: str, interval: float | None = None) -> None:
        """Type a string of text. When ``interval`` > 0 a delay is inserted
        between characters (useful for slow/legacy UIs)."""
        delay = self._type_interval if interval is None else max(0.0, interval)
        logger.debug("Typing %d character(s)", len(text))
        if delay <= 0:
            self._kb.type(text)
            return
        for ch in text:
            self._kb.type(ch)
            time.sleep(delay)

    def tap(self, key: str, presses: int = 1, interval: float | None = None) -> None:
        """Press and release a single key one or more times."""
        resolved = self._resolve_key(key)
        delay = self._key_interval if interval is None else max(0.0, interval)
        for _ in range(max(1, presses)):
            self._kb.press(resolved)
            self._kb.release(resolved)
            if delay:
                time.sleep(delay)
        logger.debug("Tapped %s x%d", key, presses)

    def press_hotkey(self, combo: str, repeats: int = 1) -> None:
        """Emit a key combination such as ``"ctrl+shift+s"``.

        All keys are pressed left-to-right and released in reverse order — the
        correct sequence for chord shortcuts on Windows. ``repeats`` re-emits
        the whole combination N times.
        """
        tokens = [t.strip() for t in combo.split("+") if t.strip()]
        if not tokens:
            raise InputSimulationError(f"Invalid hotkey combo: {combo!r}")
        keys = [self._resolve_key(t) for t in tokens]
        for _ in range(max(1, repeats)):
            for k in keys:
                self._kb.press(k)
            for k in reversed(keys):
                self._kb.release(k)
        logger.debug("Hotkey %s x%d", combo, repeats)

    def press(self, key: str) -> None:
        """Hold a key down (pair with :meth:`release`)."""
        self._kb.press(self._resolve_key(key))

    def release(self, key: str) -> None:
        """Release a previously held key."""
        self._kb.release(self._resolve_key(key))


__all__ = ["KeyboardController"]
