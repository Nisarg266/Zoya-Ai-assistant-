"""Async retry with exponential backoff + full jitter.

No external dependency — just :mod:`asyncio`. The policy is deliberately
simple and predictable:

    delay = uniform(0, min(max_delay, base_delay * factor ** attempt))

* **Exponential growth** (``factor``, default 2) gives the provider time to
  recover.
* **Full jitter** (randomising inside the window) spreads retries from many
  clients so they don't synchronously hammer the API on a 429.
* **``Retry-After`` honoured** — when the mapped exception carries a
  ``retry_after`` hint (set by :mod:`zoya.llm.errors` from the response
  header), that value is used instead, clamped to ``max_delay``.
* **Only retryable errors repeat** — ``is_retryable`` (rate-limit / server /
  timeout) gates the loop; auth or validation errors raise immediately.

Callers pass a *coroutine factory* (so each attempt creates a fresh coroutine)
plus an optional ``on_retry`` callback for logging.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from zoya.llm.errors import is_retryable, map_sdk_error

T = TypeVar("T")

#: Optional async callback: ``(attempt, delay, exception) -> None``.
RetryCallback = Callable[[int, float, BaseException], Awaitable[None] | None]


async def with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    factor: float = 2.0,
    timeout: float | None = None,
    on_retry: RetryCallback | None = None,
) -> T:
    """Run ``coro_factory()`` with retries.

    Parameters
    ----------
    coro_factory:
        Zero-arg callable returning a fresh awaitable each attempt.
    attempts:
        Total attempts including the first (must be >= 1).
    base_delay:
        Seconds — the first retry's nominal backoff before jitter.
    max_delay:
        Seconds — hard cap on any single delay (applied before jitter).
    factor:
        Multiplier applied to the delay after each failure (default 2).
    timeout:
        Per-attempt wall-clock timeout (seconds). Raises a mapped
        :class:`LLMTimeoutError` if an attempt overruns.
    on_retry:
        Optional callback invoked *before* each sleep (not before the final
        failure or the first attempt).

    Raises
    ------
    zoya.core.exceptions.LLMError
        The last (mapped) error if all attempts fail or an unretryable error
        occurs on the first attempt.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            if timeout is not None:
                return await asyncio.wait_for(coro_factory(), timeout=timeout)
            return await coro_factory()
        except BaseException as exc:  # noqa: BLE001 - mapped precisely below
            last_exc = exc
            # Always map to the LLM hierarchy before surfacing / retrying so the
            # caller never sees a raw SDK error.
            mapped = (
                exc if isinstance(exc, Exception) and _is_llm_error(exc)
                else map_sdk_error(exc)
            )

            # No attempts left, or not worth retrying → raise the mapped error.
            if attempt >= attempts or not is_retryable(mapped):
                raise mapped from exc  # type: ignore[misc]

            # Compute the backoff.
            delay = _compute_delay(
                attempt=attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                factor=factor,
                hint=_retry_after_hint(mapped),
            )

            if on_retry is not None:
                result = on_retry(attempt, delay, mapped)
                if asyncio.iscoroutine(result):
                    await result

            await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises. Kept for type checkers.
    assert last_exc is not None  # pragma: no cover
    raise map_sdk_error(last_exc)  # type: ignore[misc]  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_delay(
    *,
    attempt: int,
    base_delay: float,
    max_delay: float,
    factor: float,
    hint: float | None,
) -> float:
    """Nominal exponential backoff, jittered, honouring a server hint."""
    if hint is not None:
        return max(0.0, min(hint, max_delay))
    # attempt is 1-based and we already failed once -> use (attempt-1) as the
    # exponent so the first retry's nominal delay is ~base_delay.
    nominal = base_delay * (factor ** (attempt - 1))
    capped = min(nominal, max_delay)
    # Full jitter: uniform in [0, capped].
    return random.uniform(0.0, capped) if capped > 0 else 0.0


def _retry_after_hint(exc: BaseException) -> float | None:
    """Pull a ``retry_after`` value off a mapped exception, if present."""
    retry_after = getattr(exc, "context", {}).get("retry_after")
    if isinstance(retry_after, (int, float)):
        return float(retry_after)
    return None


def _is_llm_error(exc: BaseException) -> bool:
    """``True`` if ``exc`` is already one of our mapped LLM exceptions."""
    # Imported here to avoid a circular import with zoya.core at module load.
    from zoya.core.exceptions import LLMError

    return isinstance(exc, LLMError)


__all__ = ["with_retry", "RetryCallback"]
