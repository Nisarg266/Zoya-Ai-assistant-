"""Low-level async Gemini client wrapper.

A thin, focused adapter over ``google-genai`` 2.x:

* builds the SDK client + :class:`~google.genai.types.GenerateContentConfig`
  from :class:`~zoya.core.config.ZoyaSettings`,
* wraps every network call in :func:`zoya.llm.retry.with_retry` so transient
  429 / 5xx / timeout errors back off automatically,
* maps any raw SDK exception onto the :class:`~zoya.core.exceptions.LLMError`
  hierarchy via :mod:`zoya.llm.errors`,
* offers both a one-shot :meth:`generate` and a :meth:`generate_stream` that
  yields incremental responses.

It deliberately does **not** know about Zoya tools, history or the ReAct loop —
that orchestration lives in :class:`~zoya.llm.facade.ZoyaBrain`. This keeps the
client unit-testable with a fake ``aio.models``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from zoya.core.config import ZoyaSettings
from zoya.core.exceptions import LLMAuthError
from zoya.core.logging import get_logger
from zoya.llm.retry import with_retry

_log = get_logger("llm.client")

#: Default system persona — overridable via ZoyaBrain.
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Zoya, a highly advanced desktop AI assistant inspired by JARVIS. "
    "You run on Windows and have access to tools that can automate the system, "
    "manage files, windows, and processes. Be concise, professional, and helpful. "
    "When asked to perform an action, use the tools provided. If a tool fails, "
    "explain the error to the user."
)


class GeminiClient:
    """Async wrapper around the ``google-genai`` SDK for Zoya.

    The underlying ``genai.Client`` is created lazily on first use, so simply
    *constructing* the wrapper never needs a network or a valid key — handy for
    import-time wiring and tests.
    """

    def __init__(self, settings: ZoyaSettings) -> None:
        self._settings = settings
        self.api_key: str = settings.app.gemini_api_key.strip()
        self.model_name: str = settings.app.gemini_model
        self.api_version: str = settings.app.gemini_api_version
        self._client: genai.Client | None = None

        if not self.is_configured:
            _log.warning(
                "GEMINI_API_KEY is not set; the Brain will reject calls."
            )

    # ------------------------------------------------------------------ state
    @property
    def is_configured(self) -> bool:
        """``True`` when a non-empty API key is present."""
        return bool(self.api_key)

    def _ensure_client(self) -> genai.Client:
        """Create the SDK client on first use (raises if unconfigured)."""
        if not self.is_configured:
            raise LLMAuthError(
                "GEMINI_API_KEY is not configured.",
                code="LLM_AUTH_MISSING_KEY",
            )
        if self._client is None:
            # Disable the SDK's *internal* tenacity retry (attempts=1) so our
            # zoya.llm.retry.with_retry is the single retry authority. This keeps
            # backoff policy, logging and error mapping in one place and avoids
            # surprising double-retry latency.
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    api_version=self.api_version,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    # ------------------------------------------------------------------ config
    def _build_config(
        self,
        *,
        tools: types.Tool | None,
        system_instruction: str | None,
    ) -> types.GenerateContentConfig:
        """Assemble the per-request config from ``LLMSettings``."""
        llm = self._settings.llm
        cfg: dict[str, Any] = {
            "system_instruction": system_instruction or DEFAULT_SYSTEM_INSTRUCTION,
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "max_output_tokens": llm.max_output_tokens,
        }
        if llm.top_k and llm.top_k > 0:
            cfg["top_k"] = llm.top_k
        if tools is not None:
            cfg["tools"] = [tools]
        # Explicit thinking budget only when the user pinned one (>= 0).
        # thinking_budget == -1 means "let the model decide" → omit entirely.
        if llm.thinking_budget >= 0:
            cfg["thinking_config"] = types.ThinkingConfig(
                thinking_budget=llm.thinking_budget
            )
        return types.GenerateContentConfig(**cfg)

    # ----------------------------------------------------------- non-streaming
    async def generate(
        self,
        contents: list[types.Content],
        *,
        tools: types.Tool | None = None,
        system_instruction: str | None = None,
    ) -> types.GenerateContentResponse:
        """Send the conversation and return the full response (retry-wrapped).

        Raises:
            zoya.core.exceptions.LLMError: any mapped transport / provider error.
        """
        client = self._ensure_client()
        config = self._build_config(tools=tools, system_instruction=system_instruction)
        app = self._settings.app

        def _on_retry(attempt: int, delay: float, exc: BaseException) -> None:
            _log.warning(
                "Gemini call failed (attempt %d); retrying in %.2fs: %s",
                attempt, delay, exc,
            )

        return await with_retry(
            lambda: client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            ),
            attempts=app.llm_retry_attempts,
            base_delay=app.llm_retry_base_delay,
            max_delay=app.llm_retry_max_delay,
            timeout=app.llm_timeout,
            on_retry=_on_retry,
        )

    # --------------------------------------------------------------- streaming
    async def generate_stream(
        self,
        contents: list[types.Content],
        *,
        tools: types.Tool | None = None,
        system_instruction: str | None = None,
    ) -> AsyncIterator[types.GenerateContentResponse]:
        """Stream incremental responses as an async generator.

        The retry/backoff protects **establishment**: we create the stream *and*
        pull its first chunk inside the retry window. This matters because the
        SDK defers the real HTTP request to the first ``__anext__`` — so a 429 /
        5xx surfaces on that first chunk, which we want covered. After the first
        chunk the rest is yielded straight through; mid-stream errors are rare
        and re-running a partially consumed stream is the caller's concern.
        """
        client = self._ensure_client()
        config = self._build_config(tools=tools, system_instruction=system_instruction)
        app = self._settings.app

        def _on_retry(attempt: int, delay: float, exc: BaseException) -> None:
            _log.warning(
                "Gemini stream setup failed (attempt %d); retrying in %.2fs: %s",
                attempt, delay, exc,
            )

        async def _establish() -> tuple[
            AsyncIterator[types.GenerateContentResponse],
            types.GenerateContentResponse | None,
        ]:
            stream = await client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            try:
                first = await stream.__anext__()
            except StopAsyncIteration:
                return stream, None  # empty stream
            return stream, first

        stream, first = await with_retry(
            _establish,
            attempts=app.llm_retry_attempts,
            base_delay=app.llm_retry_base_delay,
            max_delay=app.llm_retry_max_delay,
            timeout=app.llm_timeout,
            on_retry=_on_retry,
        )

        if first is not None:
            yield first
        async for chunk in stream:
            yield chunk

    # ----------------------------------------------------------------- tokens
    async def count_tokens(self, contents: list[types.Content]) -> int:
        """Return the model's token count for ``contents``.

        Best-effort: if the ``count_tokens`` endpoint is unavailable (e.g. quota
        or network), falls back to the cheap character-based
        :meth:`estimate_tokens` so trimming still works.
        """
        client = self._ensure_client()
        try:
            resp = await client.aio.models.count_tokens(
                model=self.model_name, contents=contents
            )
            total = getattr(resp, "total_tokens", None)
            if isinstance(total, int) and total >= 0:
                return total
        except Exception:  # pragma: no cover - best-effort, degrade gracefully
            _log.debug("count_tokens failed; using char estimate.", exc_info=True)
        return self.estimate_tokens(contents)

    @staticmethod
    def estimate_tokens(contents: list[types.Content]) -> int:
        """Cheap offline token estimate (~4 chars/token) + a fixed cost per
        structured part. Good enough to drive trimming without a network call.
        """
        total_chars = 0
        structured = 0
        for content in contents:
            for part in content.parts or []:
                if part.text:
                    total_chars += len(part.text)
                if part.function_call or part.function_response:
                    structured += 1
        return (total_chars // 4) + (structured * 16)

    # ------------------------------------------------------------------ cleanup
    async def aclose(self) -> None:
        """Release the underlying async transport (idempotent)."""
        if self._client is not None:
            try:
                await self._client.aio.aclose()
            except Exception:  # pragma: no cover - best-effort teardown
                _log.debug("Ignoring error while closing Gemini client.", exc_info=True)
            finally:
                self._client = None


__all__ = ["GeminiClient", "DEFAULT_SYSTEM_INSTRUCTION"]
