"""Translate raw ``google-genai`` SDK errors into Zoya's :class:`LLMError`
hierarchy.

The Brain and client should never let a bare SDK exception bubble up: callers
want to branch on *semantics* ("am I rate-limited?" / "is my key wrong?") rather
than parse HTTP payloads. This module is the single place that translation
happens, so the rest of the codebase depends only on
:mod:`zoya.core.exceptions`.

Mapping table
-------------
============  ===================================================  =========================
SDK signal    Meaning                                             Zoya exception
============  ===================================================  =========================
429           Quota / rate limited (retryable)                    :class:`LLMRateLimitError`
5xx           Provider outage (retryable)                         :class:`LLMConnectionError`
401 / 403     Bad / missing API key (fatal)                       :class:`LLMAuthError`
other 4xx     Malformed request / blocked (fatal)                 :class:`LLMResponseError`
timeout       Request exceeded the allowed time (retryable)       :class:`LLMTimeoutError`
unknown       Anything else                                       :class:`LLMError`
============  ===================================================  =========================

The :func:`is_retryable` predicate lets :mod:`zoya.llm.retry` decide whether to
back off and try again without re-importing SDK types.
"""

from __future__ import annotations

from typing import Any

from zoya.core.exceptions import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

# Imported lazily-safe: the SDK is a hard runtime dependency for the LLM layer.
try:  # pragma: no cover - import guard only
    from google.genai import errors as _genai_errors
except Exception:  # pragma: no cover - SDK missing at type-check time only
    _genai_errors = None  # type: ignore[assignment]


# HTTP status codes that mean "the provider is overwhelmed / down".
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# HTTP status codes that mean "your credentials are wrong".
_AUTH_STATUS = {401, 403}


def _status_from(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an SDK exception."""
    # The google-genai APIError stores the code on ``.code``.
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    # Some transport errors nest the underlying response.
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after(exc: BaseException) -> float | None:
    """Return a ``Retry-After`` hint (seconds) if the SDK response carries one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        getter = headers.get  # type: ignore[union-attr]
    except AttributeError:
        return None
    raw = getter("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def map_sdk_error(exc: BaseException, *, context: dict[str, Any] | None = None) -> LLMError:
    """Convert any exception into a specific :class:`LLMError`.

    The original exception is chained as ``__cause__`` and the HTTP status (when
    available) is added to ``context`` so logs and error responses stay rich.
    """
    ctx = dict(context or {})
    status = _status_from(exc)
    if status is not None:
        ctx.setdefault("status", status)

    # --- async timeout --------------------------------------------------
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError(
            "Gemini request timed out.",
            code="LLM_TIMEOUT",
            context=ctx,
            cause=exc,
        )

    # --- not an SDK error we recognise ---------------------------------
    if _genai_errors is None or not isinstance(exc, _genai_errors.APIError):
        msg = str(exc) or type(exc).__name__
        return LLMError(
            f"Unexpected error talking to Gemini: {msg}",
            code="LLM_UNKNOWN",
            context=ctx,
            cause=exc,
        )

    # --- recognised SDK error: map by status code ----------------------
    if status in _AUTH_STATUS:
        return LLMAuthError(
            "Gemini authentication failed (check GEMINI_API_KEY).",
            code="LLM_AUTH",
            context=ctx,
            cause=exc,
        )
    if status == 429:
        ctx.setdefault("retry_after", _retry_after(exc))
        return LLMRateLimitError(
            "Gemini rate limit / quota exceeded.",
            code="LLM_RATE_LIMIT",
            context=ctx,
            cause=exc,
        )
    if status in _RETRYABLE_STATUS:
        return LLMConnectionError(
            f"Gemini server error ({status}).",
            code="LLM_CONNECTION",
            context=ctx,
            cause=exc,
        )

    # Any other client error (400, 404, blocked content, ...).
    return LLMResponseError(
        f"Gemini rejected the request ({status}).",
        code="LLM_BAD_RESPONSE",
        context=ctx,
        cause=exc,
    )


def is_retryable(exc: BaseException) -> bool:
    """``True`` if ``exc`` represents a transient, worth-retrying failure.

    Used by :func:`zoya.llm.retry.with_retry` to decide whether to back off.
    """
    if isinstance(exc, (LLMRateLimitError, LLMConnectionError, LLMTimeoutError)):
        return True
    # A raw SDK exception that we have not yet mapped: peek at its status.
    status = _status_from(exc)
    return status in _RETRYABLE_STATUS


__all__ = ["map_sdk_error", "is_retryable"]
