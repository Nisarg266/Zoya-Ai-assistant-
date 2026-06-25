"""Centralised, production-grade exception hierarchy for Zoya.

Design philosophy
-----------------
* **One root** (``ZoyaError``) — callers can catch *anything* Zoya raised with a
  single ``except ZoyaError``.
* **Subsystem mirrors** — sub-hierarchies mirror the subsystems
  (``AutomationError`` / ``ToolError`` / ``LLMError``) so callers can be as
  specific or generic as they like.
* **Rich context** — every exception optionally carries:
    * ``code``     a stable, machine-readable error code (e.g. ``"CFG_NO_KEY"``)
    * ``context``  a free-form dict of diagnostic values
  Both are rendered into the message and exposed as attributes, which makes
  logs, JSON error responses and unit assertions trivial.
* **Cause chaining** — the constructor forwards to ``Exception`` so ``raise X
  from cause`` works as expected; we never swallow the original error.

All names below are part of the public API; renaming any of them is a
breaking change for downstream modules (automation controllers, LLM facade,
tools, ...).
"""

from __future__ import annotations

from typing import Any


class ZoyaError(Exception):
    """Root error for *everything* raised inside the Zoya codebase.

    Parameters
    ----------
    message:
        Human-readable description.
    code:
        Optional stable identifier (UPPER_SNAKE_CASE) usable by clients/tests
        to branch on without parsing the message text.
    context:
        Optional diagnostic mapping merged into the rendered message. Anything
        JSON-serialisable is fine.
    cause:
        Optional original exception kept for ``raise ... from cause`` chains.
    """

    #: default code used when none is supplied by a subclass / caller
    default_code: str = "ZOYA_ERROR"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code: str = code or self.default_code
        self.context: dict[str, Any] = dict(context or {})
        self.cause: BaseException | None = cause

        rendered = self._render(message)
        super().__init__(rendered)
        if cause is not None:
            self.__cause__ = cause

    # ------------------------------------------------------------------ utils
    def _render(self, message: str) -> str:
        """Build the final human-readable message, appending context if any."""
        parts: list[str] = []
        if message:
            parts.append(message)
        if self.context:
            kv = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"[{kv}]")
        if len(parts) == 1 and self.code != self.default_code:
            # no context, but a meaningful code -> annotate the message
            return f"{parts[0]} ({self.code})"
        return " ".join(parts) if parts else self.code

    def to_dict(self) -> dict[str, Any]:
        """Serialise the error to a plain dict (great for JSON APIs / logs)."""
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "context": self.context,
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(code={self.code!r})"


# ===========================================================================
# Configuration subsystem
# ===========================================================================
class ConfigurationError(ZoyaError):
    """Configuration is missing, malformed or violates a constraint."""

    default_code = "CFG_ERROR"


# ===========================================================================
# Automation subsystem
# ===========================================================================
class AutomationError(ZoyaError):
    """Base class for all desktop-automation failures."""

    default_code = "AUTO_ERROR"


class AutomationDisabledError(AutomationError):
    """An action was requested while automation is globally disabled."""

    default_code = "AUTO_DISABLED"


class InputSimulationError(AutomationError):
    """Keyboard / mouse input could not be simulated."""

    default_code = "AUTO_INPUT"


class WindowNotFoundError(AutomationError):
    """No window matched the requested title/handle."""

    default_code = "AUTO_WINDOW_NOT_FOUND"


class ProcessError(AutomationError):
    """Process launch / query / termination failed."""

    default_code = "AUTO_PROCESS"


class FileSystemError(AutomationError):
    """A file-system operation failed (missing path, permissions, ...)."""

    default_code = "AUTO_FS"


class SystemControlError(AutomationError):
    """A system control (volume, brightness, power, ...) failed."""

    default_code = "AUTO_SYSTEM"


# ===========================================================================
# Tool / plugin layer
#
# Tool-layer problems are *not* execution crashes:
#   * validation error  -> caller bug (bad args from the LLM)
#   * not found         -> registry/lookup bug
#   * execution error   -> the tool ran but reported a *domain* failure
# ===========================================================================
class ToolError(ZoyaError):
    """Base class for tool-layer problems (distinct from automation crashes)."""

    default_code = "TOOL_ERROR"


class ToolNotFoundError(ToolError):
    """The requested tool name is not registered."""

    default_code = "TOOL_NOT_FOUND"


class ToolValidationError(ToolError):
    """Parameters supplied to a tool failed Pydantic validation."""

    default_code = "TOOL_VALIDATION"


class ToolExecutionError(ToolError):
    """A tool completed but reported a domain-level failure in its result."""

    default_code = "TOOL_EXECUTION"


# ===========================================================================
# LLM subsystem
#
# Used by the Gemini client / Brain facade. Keeping these separate from
# Automation/Tool errors lets the chat loop decide whether to retry, surface
# the error to the user, or abort.
# ===========================================================================
class LLMError(ZoyaError):
    """Base class for all Large-Language-Model failures."""

    default_code = "LLM_ERROR"


class LLMAuthError(LLMError):
    """Authentication failed — usually a missing / invalid API key."""

    default_code = "LLM_AUTH"


class LLMRateLimitError(LLMError):
    """The provider returned a rate-limit / quota error (retryable)."""

    default_code = "LLM_RATE_LIMIT"


class LLMConnectionError(LLMError):
    """A network / transport failure talking to the provider (retryable)."""

    default_code = "LLM_CONNECTION"


class LLMResponseError(LLMError):
    """The provider responded, but the payload was empty or malformed."""

    default_code = "LLM_BAD_RESPONSE"


class LLMTimeoutError(LLMError):
    """The request to the provider did not complete in the allowed time."""

    default_code = "LLM_TIMEOUT"


# ===========================================================================
# Bootstrap / lifecycle
# ===========================================================================
class BootstrapError(ZoyaError):
    """Fatal problem during application startup / shutdown."""

    default_code = "BOOTSTRAP"


__all__ = [
    # root
    "ZoyaError",
    # configuration
    "ConfigurationError",
    # automation
    "AutomationError",
    "AutomationDisabledError",
    "InputSimulationError",
    "WindowNotFoundError",
    "ProcessError",
    "FileSystemError",
    "SystemControlError",
    # tool layer
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    # llm layer
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMTimeoutError",
    # lifecycle
    "BootstrapError",
]
