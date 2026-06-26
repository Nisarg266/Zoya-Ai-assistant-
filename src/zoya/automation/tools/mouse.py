"""``mouse`` tool — cursor movement, clicking, dragging and scrolling via the LLM.

Thin adapter that exposes :class:`~zoya.automation.controllers.mouse.MouseController`
behind the :class:`~zoya.automation.tools.base.ITool` contract. A single tool
with an ``action`` enum keeps the LLM's tool list small while covering all six
operations:

* ``move``         — move to absolute ``(x, y)`` pixels
* ``click``        — click at the current position (button / clicks configurable)
* ``double_click`` — double-click at the current position
* ``right_click``  — right-click at the current position
* ``drag``         — move to ``(x, y)`` while holding a button
* ``scroll``       — scroll the wheel by ``(dx, dy)``

All validation, threading, logging and error normalisation are inherited from
:class:`~zoya.automation.tools.base.BaseTool`; this class only implements
:meth:`MouseTool._run`.
"""

from __future__ import annotations

from typing import Any

from zoya.automation.controllers.mouse import MouseController
from zoya.automation.schemas import MouseParams
from zoya.automation.tools.base import BaseTool


class MouseTool(BaseTool):
    """Simulate mouse input through one ``action``-driven entry point.

    The injected :class:`MouseController` is the Dependency-Inversion seam:
    tests pass a fake, production wiring lives only in
    :func:`~zoya.automation.tools.registry.create_default_registry`.
    """

    #: Stable, machine-friendly identifier used for dispatch & registry lookup.
    name: str = "mouse"
    #: Human-readable summary shown to the LLM alongside the JSON schema.
    description: str = (
        "Simulate mouse input. Specify an 'action': move (to absolute x,y pixels), "
        "click (at the current position), double_click, right_click, drag (to x,y "
        "while holding a button), or scroll (by dx,dy)."
    )
    #: Sends real input to the OS — not a read-only operation.
    readonly: bool = False
    #: Pydantic model that drives both schema() and parameter validation.
    ParamsModel = MouseParams

    def __init__(self, mouse: MouseController) -> None:
        # Sets up ``self._log`` under the ``zoya.automation.tools.mouse`` logger.
        super().__init__()
        self._mouse: MouseController = mouse

    # ------------------------------------------------------------------ #
    # The one method a BaseTool subclass must implement                   #
    # ------------------------------------------------------------------ #
    def _run(self, p: MouseParams) -> dict[str, Any]:
        """Dispatch ``p.action`` to the controller and return a payload.

        Domain failures (e.g. an unknown button) raise
        :class:`~zoya.core.exceptions.InputSimulationError`, which
        ``BaseTool.execute`` converts into a ``ToolResult(success=False)``.
        """
        action = p.action

        if action == "move":
            assert p.x is not None and p.y is not None  # validated by MouseParams
            smooth = True if p.smooth is None else p.smooth
            self._mouse.move(p.x, p.y, duration=p.duration, smooth=smooth)
            return {"action": action, "x": p.x, "y": p.y, "smooth": smooth}

        if action == "click":
            button = p.button or "left"
            clicks = p.clicks if p.clicks is not None else 1
            self._mouse.click(button=button, clicks=clicks, interval=p.interval)
            return {"action": action, "button": button, "clicks": clicks}

        if action == "double_click":
            button = p.button or "left"
            self._mouse.double_click(button=button)
            return {"action": action, "button": button}

        if action == "right_click":
            self._mouse.right_click()
            return {"action": action}

        if action == "drag":
            assert p.x is not None and p.y is not None
            button = p.button or "left"
            duration = p.duration if p.duration is not None else 0.5
            self._mouse.drag(p.x, p.y, duration=duration, button=button)
            return {"action": action, "x": p.x, "y": p.y, "button": button}

        # action == "scroll"
        dx = p.dx if p.dx is not None else 0
        dy = p.dy if p.dy is not None else 0
        self._mouse.scroll(dx, dy)
        return {"action": action, "dx": dx, "dy": dy}


__all__ = ["MouseTool"]
