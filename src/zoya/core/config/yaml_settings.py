"""YAML-driven tunable settings.

Where :mod:`zoya.core.config.env` holds *secrets & flags*, this module holds
*tunable defaults*: values that are identical across machines, are safe to
commit, and are meant to be hand-edited by the developer in
``config/settings.yaml``.

Each model is a plain :class:`pydantic.BaseModel` with ``Field`` constraints,
so a malformed value in the YAML file surfaces as a precise, attributed
validation error at load time instead of a mysterious crash deep in a
controller.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AutomationSettings(BaseModel):
    """Tunable automation defaults (loaded from ``settings.yaml``)."""

    model_config = {"extra": "forbid"}

    default_type_interval: float = Field(
        0.0, ge=0, description="Seconds between characters when typing."
    )
    key_press_interval: float = Field(
        0.1, ge=0, description="Seconds between repeated key taps."
    )
    mouse_move_duration: float = Field(
        0.3, ge=0, description="Seconds for a smooth cursor move."
    )
    mouse_move_steps: int = Field(
        50, ge=1, description="Interpolation steps per smooth move."
    )
    click_interval: float = Field(
        0.1, ge=0, description="Seconds between successive clicks."
    )
    scroll_amount: int = Field(3, ge=1, description="Default scroll magnitude.")
    launch_timeout: float = Field(
        10.0, ge=0, description="Seconds to wait when launching an app."
    )
    screenshot_dir: str = "screenshots"
    safe_delete: bool = Field(
        True, description="Prefer the recycle bin over permanent deletion."
    )


class PathSettings(BaseModel):
    """Default on-disk locations used by Zoya (relative to the project root)."""

    model_config = {"extra": "forbid"}

    notes_dir: str = "data/notes"


class LLMSettings(BaseModel):
    """Generation + behaviour tunables for the Gemini Brain.

    Loaded from the ``llm:`` block of ``settings.yaml``. These describe *how*
    the model should answer and how the ReAct loop is bounded; transport-level
    knobs (timeout, retry) live on :class:`~zoya.core.config.AppConfig` because
    they're more operational than behavioural.
    """

    model_config = {"extra": "forbid"}

    #: Sampling temperature (0 = deterministic, 1 = creative, 2 = max).
    temperature: float = Field(0.7, ge=0, le=2)
    #: Nucleus sampling probability mass.
    top_p: float = Field(1.0, ge=0, le=1)
    #: Top-K tokens considered at each step (0 = provider default).
    top_k: int = Field(0, ge=0)
    #: Hard cap on the number of tokens in a single model response.
    max_output_tokens: int = Field(8192, ge=1)
    #: Gemini-2.5 thinking budget in tokens. ``-1`` = dynamic (model-decided),
    #: ``0`` = disable thinking. Ignored by models without a thinking mode.
    thinking_budget: int = Field(-1, ge=-1)
    #: Max ReAct iterations (user prompt -> tool calls -> answer) before the
    #: Brain aborts to prevent runaway tool ping-pong.
    max_react_iterations: int = Field(8, ge=1)
    #: Hard cap on the number of user/model turn-pairs kept in memory.
    #: Oldest pairs are dropped first (never splitting a tool-call from its
    #: response). 0 = rely solely on the token budget below.
    max_history_turns: int = Field(20, ge=0)
    #: Soft token budget for conversation history. Before each model call the
    #: Brain counts tokens and trims oldest turn-pairs until under budget, so a
    #: long chat never overflows the model's context window. 0 = disable
    #: token-based trimming (use only the turn cap above).
    max_history_tokens: int = Field(50_000, ge=0)


__all__ = ["AutomationSettings", "PathSettings", "LLMSettings"]
