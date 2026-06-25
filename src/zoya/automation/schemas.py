"""Pydantic schemas for the Desktop Automation tool/plugin layer.

This module is intentionally free of any ``zoya.*`` imports so it can be used
independently (e.g. to generate JSON Schemas for an LLM) without pulling in the
controllers or Windows-only dependencies.

Two kinds of objects live here:

* **``ToolParams``** — a common base for every tool's parameter model. It forbids
  unknown fields (``extra="forbid"``) so that a hallucinated LLM argument fails
  fast with a clear validation error instead of being silently ignored.
* One concrete ``XxxParams`` model per tool. The model's ``model_json_schema()``
  is exactly what becomes a Gemini *function declaration* in a later module, so
  every field carries a human-readable ``description`` and sensible constraints.
* **``ToolResult``** — the uniform return type of every tool. Pydantic-based so
  it serialises cleanly back to the LLM / UI as JSON.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Shared parameter base
# ---------------------------------------------------------------------------
class ToolParams(BaseModel):
    """Base for all tool parameter models.

    ``extra="forbid"`` turns typos / hallucinated arguments into a
    :class:`~zoya.core.exceptions.ToolValidationError` at the boundary, which is
    far safer than silently dropping them.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class ToolResult(BaseModel):
    """Uniform outcome of a tool execution.

    A tool either succeeds (``success=True`` + a ``data`` payload) or reports a
    domain-level failure (``success=False`` + an ``error`` message and the
    exception class name in ``error_type``). Validation failures are NOT
    represented here: they are raised as :class:`ToolValidationError` because
    they indicate a caller bug, not a runtime/environment problem.
    """

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="Whether the tool completed without error.")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload returned by the tool on success.",
    )
    error: Optional[str] = Field(
        default=None, description="Human-readable error message when success is False."
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Class name of the underlying exception (for diagnostics).",
    )

    @property
    def ok(self) -> bool:
        """Convenience alias for ``success``."""
        return self.success

    def to_payload(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON / LLM consumption."""
        return self.model_dump()


# ===========================================================================
# Keyboard parameters
# ===========================================================================
class TypeTextParams(ToolParams):
    text: str = Field(..., min_length=1, description="The text to type character by character.")
    interval: float = Field(0.0, ge=0, description="Seconds of delay between characters.")


class TapKeyParams(ToolParams):
    key: str = Field(
        ...,
        min_length=1,
        description="A key token, e.g. 'enter', 'ctrl', 'f5', or a single character.",
    )
    presses: int = Field(1, ge=1, description="Number of times to press and release the key.")
    interval: float = Field(0.1, ge=0, description="Seconds between repeated presses.")


class PressHotkeyParams(ToolParams):
    combo: str = Field(
        ...,
        min_length=1,
        description="A key chord joined by '+', e.g. 'ctrl+shift+s' or 'alt+tab'.",
    )
    repeats: int = Field(1, ge=1, description="Number of times to emit the whole chord.")


# ===========================================================================
# Mouse parameters
# ===========================================================================
class MouseMoveParams(ToolParams):
    x: int = Field(..., description="Target absolute screen X coordinate (pixels).")
    y: int = Field(..., description="Target absolute screen Y coordinate (pixels).")
    smooth: bool = Field(True, description="If True, interpolate movement (looks human).")
    duration: float = Field(0.3, ge=0, description="Seconds the smooth move should take.")


class MouseClickParams(ToolParams):
    button: Literal["left", "right", "middle"] = Field("left", description="Which mouse button.")
    clicks: int = Field(1, ge=1, description="Number of clicks at the current cursor position.")
    interval: float = Field(0.1, ge=0, description="Seconds between successive clicks.")


class MouseScrollParams(ToolParams):
    dx: int = Field(0, description="Horizontal scroll amount (positive = right).")
    dy: int = Field(0, description="Vertical scroll amount (positive = up).")


class MousePositionParams(ToolParams):
    """No parameters — reads the current cursor location."""


class MouseDragParams(ToolParams):
    x: int = Field(..., description="Destination X coordinate for the drag.")
    y: int = Field(..., description="Destination Y coordinate for the drag.")
    button: Literal["left", "right", "middle"] = Field("left", description="Button held during drag.")
    duration: float = Field(0.5, ge=0, description="Seconds the drag movement should take.")


# ===========================================================================
# Window parameters
# ===========================================================================
class ListWindowsParams(ToolParams):
    include_empty: bool = Field(
        False, description="Include windows that have no title (usually hidden)."
    )


class FocusWindowParams(ToolParams):
    title: str = Field(..., min_length=1, description="Title (or substring) of the window to focus.")


class WindowActionParams(ToolParams):
    title: str = Field(..., min_length=1, description="Title (or substring) of the target window.")
    action: Literal["minimize", "maximize", "restore", "close"] = Field(
        ..., description="The window operation to perform."
    )


class ActiveWindowParams(ToolParams):
    """No parameters — returns the foreground window."""


# ===========================================================================
# File-system parameters
# ===========================================================================
class ListDirectoryParams(ToolParams):
    path: str = Field(..., description="Directory to list (absolute or user-relative).")
    pattern: str = Field("*", description="Glob pattern each entry must match.")


class SearchFilesParams(ToolParams):
    directory: str = Field(..., description="Root directory to search under.")
    pattern: str = Field("**/*", description="Glob pattern; supports '**' for recursion.")
    recursive: bool = Field(True, description="If False, only the top level is searched.")


class ReadFileParams(ToolParams):
    path: str = Field(..., description="Path of the text file to read.")
    encoding: str = Field("utf-8", description="Text encoding to use when reading.")


class WriteFileParams(ToolParams):
    path: str = Field(..., description="Destination file path (parents are created).")
    content: str = Field(..., description="The text to write.")
    append: bool = Field(False, description="If True, append instead of overwriting.")


class FileExistsParams(ToolParams):
    path: str = Field(..., description="Path whose existence should be checked.")


# ===========================================================================
# Process parameters
# ===========================================================================
class LaunchAppParams(ToolParams):
    target: str = Field(
        ..., min_length=1, description="App name/alias (e.g. 'notepad') or an executable path."
    )
    args: Optional[str] = Field(None, description="Command-line arguments (whitespace split).")
    working_dir: Optional[str] = Field(None, description="Working directory for the new process.")


class ListProcessesParams(ToolParams):
    name_filter: Optional[str] = Field(
        None, description="Only list processes whose name contains this substring."
    )


class TerminateProcessParams(ToolParams):
    """Terminate by name XOR pid — exactly one must be provided."""

    name: Optional[str] = Field(None, description="Process name substring to match.")
    pid: Optional[int] = Field(None, ge=1, description="Specific process ID to terminate.")
    force: bool = Field(False, description="If True, kill (SIGKILL) instead of asking politely.")

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "TerminateProcessParams":
        if (self.name is None) == (self.pid is None):
            raise ValueError("Provide exactly one of 'name' or 'pid' (not both, not neither).")
        return self


# ===========================================================================
# System parameters
# ===========================================================================
class GetVolumeParams(ToolParams):
    """No parameters — returns the current master volume."""


class SetVolumeParams(ToolParams):
    level: int = Field(..., ge=0, le=100, description="Target volume as a percentage (0-100).")


class GetBrightnessParams(ToolParams):
    """No parameters — returns the current screen brightness."""


class SetBrightnessParams(ToolParams):
    level: int = Field(..., ge=0, le=100, description="Target brightness as a percentage (0-100).")


class ClipboardGetParams(ToolParams):
    """No parameters — returns the clipboard text."""


class ClipboardSetParams(ToolParams):
    text: str = Field(..., description="Text to place on the system clipboard.")


class PowerParams(ToolParams):
    action: Literal["lock", "sleep", "shutdown", "restart", "logoff"] = Field(
        ..., description="The power action to perform."
    )
    confirm: bool = Field(
        True,
        description="Destructive actions (sleep/shutdown/restart/logoff) require confirmation.",
    )


__all__ = [
    # base + result
    "ToolParams",
    "ToolResult",
    # keyboard
    "TypeTextParams",
    "TapKeyParams",
    "PressHotkeyParams",
    # mouse
    "MouseMoveParams",
    "MouseClickParams",
    "MouseScrollParams",
    "MousePositionParams",
    "MouseDragParams",
    # window
    "ListWindowsParams",
    "FocusWindowParams",
    "WindowActionParams",
    "ActiveWindowParams",
    # filesystem
    "ListDirectoryParams",
    "SearchFilesParams",
    "ReadFileParams",
    "WriteFileParams",
    "FileExistsParams",
    # process
    "LaunchAppParams",
    "ListProcessesParams",
    "TerminateProcessParams",
    # system
    "GetVolumeParams",
    "SetVolumeParams",
    "GetBrightnessParams",
    "SetBrightnessParams",
    "ClipboardGetParams",
    "ClipboardSetParams",
    "PowerParams",
]
