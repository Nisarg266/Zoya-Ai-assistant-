"""``keyboard`` tool — typed/pressed/held key simulation via the LLM.

Thin adapter that exposes :class:`~zoya.automation.controllers.keyboard.KeyboardController`
behind the :class:`~zoya.automation.tools.base.ITool` contract. A single tool
with an ``action`` enum keeps the LLM's tool list small while still covering all
five keyboard operations:

* ``type_text``    — type a string of text
* ``press_key``    — press and release a key (optionally N times)
* ``hotkey``       — emit a chord such as ``"ctrl+shift+s"``
* ``hold_key``     — press and hold a key down
* ``release_key``  — release a previously held key

All validation, threading, logging and error normalisation are inherited from
:class:`~zoya.automation.tools.base.BaseTool`; this class only implements
:meth:`KeyboardTool._run`.
"""

from __future__ import annotations

from typing import Any

from zoya.automation.controllers.keyboard import KeyboardController
from zoya.automation.schemas import KeyboardParams
from zoya.automation.tools.base import BaseTool


class KeyboardTool(BaseTool):
    """Simulate keyboard input through one ``action``-driven entry point.

    The injected :class:`KeyboardController` is the Dependency-Inversion seam:
    tests pass a fake, production wiring lives only in
    :func:`~zoya.automation.tools.registry.create_default_registry`.
    """

    #: Stable, machine-friendly identifier used for dispatch & registry lookup.
    name: str = "keyboard"
    #: Human-readable summary shown to the LLM alongside the JSON schema.
    description: str = (
        "Simulate keyboard input. Specify an 'action': type_text (type a string), "
        "press_key (press+release a key, optionally N times), hotkey (emit a chord "
        "like 'ctrl+shift+s'), hold_key (press and hold), or release_key (release a "
        "held key)."
    )
    #: Sends real input to the OS — not a read-only operation.
    readonly: bool = False
    #: Pydantic model that drives both schema() and parameter validation.
    ParamsModel = KeyboardParams

    def __init__(self, keyboard: KeyboardController) -> None:
        # Sets up ``self._log`` under the ``zoya.automation.tools.keyboard`` logger.
        super().__init__()
        self._kb: KeyboardController = keyboard

    # ------------------------------------------------------------------ #
    # The one method a BaseTool subclass must implement                   #
    # ------------------------------------------------------------------ #
    def _run(self, p: KeyboardParams) -> dict[str, Any]:
        """Dispatch ``p.action`` to the controller and return a payload.

        Domain failures (e.g. an unknown key token) raise
        :class:`~zoya.core.exceptions.InputSimulationError`, which
        ``BaseTool.execute`` converts into a ``ToolResult(success=False)``.
        """
        action = p.action

        if action == "type_text":
            assert p.text is not None  # validated by KeyboardParams
            self._kb.type_text(p.text, interval=p.interval)
            return {"action": action, "text": p.text, "chars": len(p.text)}

        if action == "press_key":
            assert p.key is not None
            presses = p.presses if p.presses is not None else 1
            self._kb.press_key(p.key, presses=presses, interval=p.interval)
            return {"action": action, "key": p.key, "presses": presses}

        if action == "hotkey":
            assert p.combo is not None
            repeats = p.repeats if p.repeats is not None else 1
            self._kb.press_hotkey(p.combo, repeats=repeats)
            return {"action": action, "combo": p.combo, "repeats": repeats}

        if action == "hold_key":
            assert p.key is not None
            self._kb.hold_key(p.key)
            return {"action": action, "key": p.key, "held": True}

        # action == "release_key"
        assert p.key is not None
        self._kb.release_key(p.key)
        return {"action": action, "key": p.key, "held": False}


__all__ = ["KeyboardTool"]
