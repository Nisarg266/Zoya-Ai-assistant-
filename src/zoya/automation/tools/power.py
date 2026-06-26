"""``power`` tool — Windows power-state actions via the LLM.

Thin adapter that exposes :meth:`SystemController.power` behind the
:class:`~zoya.automation.tools.base.ITool` contract. Supports five actions:

* ``lock``     — lock the workstation (non-destructive, immediate)
* ``sleep``    — suspend the machine                (destructive)
* ``logoff``   — sign out the current user           (destructive)
* ``restart``  — reboot the machine                  (destructive)
* ``shutdown`` — power off the machine               (destructive)

Safety model
------------
Destructive actions are **blocked by default**. ``PowerParams.confirm`` defaults
to ``False``; the caller (the Brain, after obtaining real user confirmation)
must explicitly re-invoke with ``confirm=true`` for a destructive action to
execute. A destructive call without confirmation raises
:class:`~zoya.core.exceptions.SystemControlError`, which ``BaseTool.execute``
converts into a ``ToolResult(success=False)`` — so the ReAct loop sees a clear
"confirmation required" signal and can ask the user. ``lock`` is non-destructive
and runs regardless of ``confirm``.

All validation, threading, logging and error normalisation are inherited from
:class:`~zoya.automation.tools.base.BaseTool`; this class only implements
:meth:`PowerTool._run`.
"""

from __future__ import annotations

from typing import Any

from zoya.automation.controllers.system import SystemController
from zoya.automation.schemas import PowerParams
from zoya.automation.tools.base import BaseTool


# Actions that can interrupt or end the user's session / lose unsaved work.
_DESTRUCTIVE_ACTIONS = frozenset({"sleep", "shutdown", "restart", "logoff"})


class PowerTool(BaseTool):
    """Perform Windows power-state actions with mandatory confirmation.

    The injected :class:`SystemController` is the Dependency-Inversion seam:
    tests pass a fake, production wiring lives only in
    :func:`~zoya.automation.tools.registry.create_default_registry`.
    """

    #: Stable, machine-friendly identifier used for dispatch & registry lookup.
    name: str = "power"
    #: Human-readable summary shown to the LLM alongside the JSON schema.
    description: str = (
        "Perform a Windows power action: lock, sleep, logoff, restart or shutdown. "
        "Destructive actions (sleep/logoff/restart/shutdown) require confirm=true; "
        "they are blocked otherwise. 'lock' runs immediately."
    )
    #: Power actions change system state — not a read-only operation.
    readonly: bool = False
    #: Pydantic model that drives both schema() and parameter validation.
    ParamsModel = PowerParams

    def __init__(self, system: SystemController) -> None:
        # Sets up ``self._log`` under the ``zoya.automation.tools.power`` logger.
        super().__init__()
        self._sys: SystemController = system

    # ------------------------------------------------------------------ #
    # The one method a BaseTool subclass must implement                   #
    # ------------------------------------------------------------------ #
    def _run(self, p: PowerParams) -> dict[str, Any]:
        """Execute the power action via the controller and return a payload.

        A destructive action without ``confirm=True`` raises
        :class:`~zoya.core.exceptions.SystemControlError` (raised inside
        :meth:`SystemController.power`), which ``BaseTool.execute`` converts
        into a ``ToolResult(success=False)``.
        """
        destructive = p.action in _DESTRUCTIVE_ACTIONS
        status = self._sys.power(p.action, confirm=p.confirm)
        return {
            "action": p.action,
            "status": status,
            "confirm": p.confirm,
            "destructive": destructive,
        }


__all__ = ["PowerTool"]
