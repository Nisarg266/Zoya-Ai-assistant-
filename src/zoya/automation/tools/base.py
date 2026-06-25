"""The ``ITool`` contract and the ``BaseTool`` boilerplate-killer.

Design goals
------------
* **Dependency Inversion**: the registry and the facade depend only on the
  :class:`ITool` *protocol*, never on concrete tools. New tools plug in without
  touching callers.
* **Single tool effort**: a concrete tool implements exactly one method —
  :meth:`BaseTool._run`. Validation, async offloading, logging and error
  normalisation are handled once, here.
* **Clear error semantics** (matching :mod:`zoya.core.exceptions`):

  - Bad parameters → raise :class:`ToolValidationError` (a *caller* bug; the
    planner/LLM should fix and retry).
  - A domain failure (any :class:`ZoyaError`) → returned as a
    :class:`ToolResult` with ``success=False`` (an *environment* problem).
  - Any other unexpected exception → also returned as a failed ``ToolResult``
    so one misbehaving tool never crashes the whole assistant.

Async model
-----------
Controllers are blocking and synchronous (Win32 / pynput are not async).
:meth:`BaseTool.execute` therefore runs the blocking ``_run`` in a worker
thread via :func:`asyncio.to_thread`, keeping the event loop responsive.
"""

from __future__ import annotations

import abc
import asyncio
from typing import Any, ClassVar, Protocol, Type, runtime_checkable

from pydantic import BaseModel, ValidationError

from zoya.automation.schemas import ToolResult
from zoya.core.exceptions import ToolValidationError, ZoyaError
from zoya.core.logging import get_logger


@runtime_checkable
class ITool(Protocol):
    """Structural contract every tool plugin satisfies.

    Implementations normally subclass :class:`BaseTool` rather than satisfying
    this protocol by hand.
    """

    #: Stable, machine-friendly identifier used for dispatch & registry lookup.
    name: str
    #: Human-readable summary shown to the LLM alongside the JSON schema.
    description: str
    #: ``True`` for read-only tools (they still execute in dry-run mode).
    readonly: bool

    def schema(self) -> dict[str, Any]:
        """Return the Gemini-style function declaration for this tool."""
        ...

    async def execute(self, params: dict[str, Any] | None = None) -> ToolResult:
        """Validate ``params`` then run the tool, returning a ToolResult."""
        ...


class BaseTool(abc.ABC):
    """Concrete helper that removes all tool boilerplate.

    Subclasses declare four class attributes and implement :meth:`_run`::

        class TypeTextTool(BaseTool):
            name = "type_text"
            description = "Type a string of text."
            readonly = False
            ParamsModel = TypeTextParams

            def __init__(self, keyboard: KeyboardController) -> None:
                super().__init__()
                self._kb = keyboard

            def _run(self, p: TypeTextParams) -> dict[str, Any]:
                self._kb.type_text(p.text, interval=p.interval)
                return {"typed": p.text}

    The injected controller instance is the Dependency-Inversion seam: tests can
    pass a fake, and the production wiring lives only in ``create_default_registry``.
    """

    # --- declared by subclasses ------------------------------------------- #
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    readonly: ClassVar[bool] = False
    ParamsModel: ClassVar[Type[BaseModel]] = BaseModel

    # ---------------------------------------------------------------------- #
    def __init__(self) -> None:
        # Each tool gets its own child logger under zoya.automation.tools.<name>.
        self._log = get_logger(f"automation.tools.{self.name}")

    @property
    def log(self):  # noqa: ANN201 - trivial getter, typed via assignment
        """This tool's logger (handy inside ``_run``)."""
        return self._log

    # ---------------------------------------------------------------------- #
    # Function-declaration / schema                                          #
    # ---------------------------------------------------------------------- #
    def schema(self) -> dict[str, Any]:
        """Build the Gemini-style function declaration from the ParamsModel.

        The Pydantic JSON schema already includes field descriptions and
        constraints (``min_length``, ``ge``/``le``, enums...), which is exactly
        what a function-calling model needs to call the tool correctly.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.ParamsModel.model_json_schema(),
        }

    # ---------------------------------------------------------------------- #
    # Async entry point                                                      #
    # ---------------------------------------------------------------------- #
    async def execute(self, params: dict[str, Any] | None = None) -> ToolResult:
        """Validate ``params``, run ``_run`` in a thread, normalise the outcome.

        - :class:`ToolValidationError` propagates (caller bug).
        - :class:`ZoyaError` → ``ToolResult(success=False)`` (domain failure).
        - Any other ``Exception`` → failed ``ToolResult`` (defensive; logged).
        """
        raw = params or {}
        try:
            validated = self.ParamsModel.model_validate(raw)
        except ValidationError as exc:
            # Caller supplied bad/unknown arguments — surface immediately.
            raise ToolValidationError(f"[{self.name}] invalid parameters: {exc}") from exc

        try:
            data = await asyncio.to_thread(self._run, validated)
        except ZoyaError as exc:
            self._log.warning("Tool %s failed: %s", self.name, exc)
            return ToolResult(
                success=False, error=str(exc), error_type=type(exc).__name__
            )
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._log.exception("Unexpected error while running tool %s", self.name)
            return ToolResult(
                success=False,
                error=f"Unexpected error: {exc}",
                error_type=type(exc).__name__,
            )

        return ToolResult(success=True, data=data or {})

    # ---------------------------------------------------------------------- #
    # The one method a subclass must implement                               #
    # ---------------------------------------------------------------------- #
    @abc.abstractmethod
    def _run(self, params: Any) -> dict[str, Any]:
        """Synchronous domain work. Return a JSON-serialisable payload dict."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Tool {self.name!r} readonly={self.readonly}>"


__all__ = ["ITool", "BaseTool"]
