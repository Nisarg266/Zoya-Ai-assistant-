"""``open_app`` tool — launch installed applications by friendly name.

Thin adapter that wraps :class:`~zoya.automation.controllers.app_controller.AppController`
behind the :class:`~zoya.automation.tools.base.ITool` contract so the Gemini
brain can open apps such as ``"notepad"`` or ``"calculator"`` using the
configurable application registry (``config/applications.yaml``).

All validation, threading, logging and error normalisation are inherited from
:class:`~zoya.automation.tools.base.BaseTool`; this class only implements
:meth:`OpenAppTool._run`.
"""

from __future__ import annotations

from typing import Any

from zoya.automation.controllers.app_controller import AppController, ProcessLaunchResult
from zoya.automation.schemas import OpenAppParams
from zoya.automation.tools.base import BaseTool


class OpenAppTool(BaseTool):
    """Launch an installed Windows application by its friendly name.

    The injected :class:`AppController` is the Dependency-Inversion seam: tests
    can pass a fake, and production wiring lives only in
    :func:`~zoya.automation.tools.registry.create_default_registry`.
    """

    #: Stable, machine-friendly identifier used for dispatch & registry lookup.
    name: str = "open_app"
    #: Human-readable summary shown to the LLM alongside the JSON schema.
    description: str = (
        "Launch an installed Windows application by its friendly name, e.g. "
        "'notepad', 'calculator', 'vscode'. Names are resolved through the "
        "configurable application registry (config/applications.yaml); unknown "
        "names are still attempted via the system PATH. Returns the resolved "
        "executable and new process id (when available)."
    )
    #: Writes/launches a process — not a read-only operation.
    readonly: bool = False
    #: Pydantic model that drives both schema() and parameter validation.
    ParamsModel = OpenAppParams

    def __init__(self, app_controller: AppController) -> None:
        # Sets up ``self._log`` under the ``zoya.automation.tools.open_app`` logger.
        super().__init__()
        self._apps: AppController = app_controller

    # ------------------------------------------------------------------ #
    # The one method a BaseTool subclass must implement                   #
    # ------------------------------------------------------------------ #
    def _run(self, p: OpenAppParams) -> dict[str, Any]:
        """Open the requested application and return a JSON-serialisable payload.

        Domain failures (``ApplicationNotFoundError`` / ``AppLaunchError``) are
        :class:`~zoya.core.exceptions.ZoyaError` subclasses; ``BaseTool.execute``
        converts them into a ``ToolResult(success=False)``.
        """
        result: ProcessLaunchResult = self._apps.open_app(
            p.name, args=p.args, working_dir=p.working_dir
        )
        return {
            "name": result.name,
            "executable": result.executable,
            "pid": result.pid,
            "args": result.args,
            "elevated": result.elevated,
            "via_registry": result.via_registry,
        }


__all__ = ["OpenAppTool"]
