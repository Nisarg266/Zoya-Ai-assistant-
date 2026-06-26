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


class KeyboardParams(ToolParams):
    """Unified parameters for the ``keyboard`` tool.

    A single ``action`` selects one of five operations; only the fields relevant
    to that action are required (enforced by :meth:`_enforce_action_fields`).
    ``extra="forbid"`` (inherited) rejects any unrecognised argument, and the
    validator turns an action/field mismatch into a clear validation error.
    """

    action: Literal["type_text", "press_key", "hotkey", "hold_key", "release_key"] = Field(
        ..., description="Which keyboard operation to perform."
    )
    text: Optional[str] = Field(
        None, description="Text to type (required for action='type_text')."
    )
    key: Optional[str] = Field(
        None,
        description=(
            "A single key token, e.g. 'enter', 'ctrl', 'f5' or a single character "
            "(required for press_key / hold_key / release_key)."
        ),
    )
    combo: Optional[str] = Field(
        None,
        description="A key chord joined by '+', e.g. 'ctrl+shift+s' (required for action='hotkey').",
    )
    interval: Optional[float] = Field(
        None,
        ge=0,
        description="Delay (seconds) between characters (type_text) or between presses (press_key).",
    )
    presses: Optional[int] = Field(
        None, ge=1, description="Number of press/release cycles (action='press_key'). Defaults to 1."
    )
    repeats: Optional[int] = Field(
        None, ge=1, description="Number of times to emit the whole chord (action='hotkey'). Defaults to 1."
    )

    @model_validator(mode="after")
    def _enforce_action_fields(self) -> "KeyboardParams":
        """Ensure the fields required by the chosen ``action`` are present."""
        a = self.action
        if a == "type_text":
            if not self.text:
                raise ValueError("action 'type_text' requires a non-empty 'text'.")
        elif a == "press_key":
            if not self.key:
                raise ValueError("action 'press_key' requires a non-empty 'key'.")
        elif a == "hotkey":
            if not self.combo:
                raise ValueError("action 'hotkey' requires a non-empty 'combo'.")
        elif a in ("hold_key", "release_key"):
            if not self.key:
                raise ValueError(f"action '{a}' requires a non-empty 'key'.")
        return self


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


class MouseParams(ToolParams):
    """Unified parameters for the ``mouse`` tool.

    A single ``action`` selects one of six operations; only the fields relevant
    to that action are required (enforced by :meth:`_enforce_action_fields`).
    ``extra="forbid"`` (inherited) rejects any unrecognised argument.

    Coordinate actions (``move``/``drag``) take absolute ``(x, y)`` pixels;
    ``click``/``double_click``/``right_click`` act at the current cursor
    position (combine with ``move`` to click a specific point).
    """

    action: Literal["move", "click", "double_click", "right_click", "drag", "scroll"] = Field(
        ..., description="Which mouse operation to perform."
    )
    x: Optional[int] = Field(None, description="Target absolute X pixel (move / drag).")
    y: Optional[int] = Field(None, description="Target absolute Y pixel (move / drag).")
    button: Optional[Literal["left", "right", "middle"]] = Field(
        None, description="Mouse button (click / double_click / drag). Defaults to 'left'."
    )
    clicks: Optional[int] = Field(
        None, ge=1, description="Number of clicks (action='click'). Defaults to 1."
    )
    interval: Optional[float] = Field(
        None, ge=0, description="Seconds between successive clicks (action='click')."
    )
    smooth: Optional[bool] = Field(
        None, description="Interpolate movement (move). Defaults to True."
    )
    duration: Optional[float] = Field(
        None, ge=0, description="Seconds the movement should take (move / drag)."
    )
    dx: Optional[int] = Field(None, description="Horizontal scroll amount (positive = right).")
    dy: Optional[int] = Field(None, description="Vertical scroll amount (positive = up).")

    @model_validator(mode="after")
    def _enforce_action_fields(self) -> "MouseParams":
        """Ensure the fields required by the chosen ``action`` are present."""
        a = self.action
        if a in ("move", "drag"):
            if self.x is None or self.y is None:
                raise ValueError(f"action {a!r} requires 'x' and 'y'.")
        elif a == "scroll":
            if self.dx is None and self.dy is None:
                raise ValueError("action 'scroll' requires 'dx' and/or 'dy'.")
        # click / double_click / right_click have no required fields.
        return self


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
# Application registry parameters
# ===========================================================================
class OpenAppParams(ToolParams):
    """Parameters for the ``open_app`` tool (registry-driven app launching).

    ``name`` is resolved through ``config/applications.yaml`` (canonical name or
    alias) and falls back to the system PATH, so a bare exe name still works.

    Arguments are a ``list[str]`` (one token per element) so values containing
    spaces — e.g. file paths — are passed to the child verbatim.
    """

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "Friendly application name as registered in config/applications.yaml "
            "(e.g. 'notepad', 'calculator', 'vscode'). Unknown names are "
            "attempted via the system PATH."
        ),
    )
    args: Optional[list[str]] = Field(
        None,
        description=(
            "Extra command-line arguments appended to the app's defaults. "
            "One token per element (e.g. ['--new-window', 'C:/path/with space.txt'])."
        ),
    )
    working_dir: Optional[str] = Field(
        None, description="Override working directory for the new process."
    )


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
        False,
        description=(
            "Must be explicitly true to execute a destructive action "
            "(sleep/shutdown/restart/logoff). Defaults to false — destructive "
            "actions are blocked until the caller re-invokes with confirm=true. "
            "'lock' is non-destructive and runs regardless."
        ),
    )


__all__ = [
    # base + result
    "ToolParams",
    "ToolResult",
    # keyboard
    "TypeTextParams",
    "TapKeyParams",
    "PressHotkeyParams",
    "KeyboardParams",
    # mouse
    "MouseMoveParams",
    "MouseClickParams",
    "MouseScrollParams",
    "MousePositionParams",
    "MouseDragParams",
    "MouseParams",
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
    # application registry
    "OpenAppParams",
    # system
    "GetVolumeParams",
    "SetVolumeParams",
    "GetBrightnessParams",
    "SetBrightnessParams",
    "ClipboardGetParams",
    "ClipboardSetParams",
    "PowerParams",
]
