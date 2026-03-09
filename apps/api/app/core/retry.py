"""Retry utilities with exponential backoff for external API calls.

Per AGENTS.md Rule 4: Never raises exceptions - returns None on exhaustion.
"""

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# HTTP status codes that are permanent failures and should never be retried.
# 4xx errors (except 429 Too Many Requests) are client errors — retrying
# will not fix them and wastes time + quota on each attempt.
_DEFAULT_NON_RETRYABLE_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 410, 422})


def _is_retryable(
    exc: BaseException,
    non_retryable_statuses: frozenset[int] = _DEFAULT_NON_RETRYABLE_STATUSES,
) -> bool:
    """Return True when the exception represents a transient/retryable failure."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in non_retryable_statuses
    # All other exceptions (network errors, timeouts, etc.) are retryable
    return True


def _log_retry_attempt(func_name: str, exc: Exception, *, attempt: int, delay_seconds: float) -> None:
    """Log one retry attempt with delay and error details."""
    status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    logger.warning(
        "retry.attempt",
        func=func_name,
        attempt=attempt,
        delay_seconds=round(delay_seconds, 2),
        status_code=status_code,
        error=str(exc),
    )


def _log_exhaustion(func_name: str, exc: Exception, *, attempts: int) -> None:
    """Return None on exhaustion and log the final failure."""
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code in _DEFAULT_NON_RETRYABLE_STATUSES:
            logger.warning(
                "retry.non_retryable_http_error",
                func=func_name,
                status_code=status_code,
                error=str(exc),
            )
            return

    logger.error(
        "retry.exhausted",
        func=func_name,
        attempts=attempts,
        error=str(exc),
    )


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    non_retryable_statuses: frozenset[int] = _DEFAULT_NON_RETRYABLE_STATUSES,
    **kwargs: Any,
) -> T | None:
    """Retry async function with exponential backoff + jitter on failure.

    Retries on transient exceptions (network errors, 5xx), with exponential
    delay plus random jitter to avoid thundering herd on recovery.
    Returns None if all retries exhausted (never raises).

    Non-retryable HTTP status codes (4xx except 429) are failed immediately
    without sleeping — retrying a 403 is pointless and wastes 3+ seconds.

    Per AGENTS.md Rule 4: Tools never raise exceptions. Callers must
    handle None return value (e.g., return [] from tool).

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_attempts: Maximum retry attempts (default 3)
        base_delay: Initial retry delay in seconds (default 1.0)
        max_delay: Maximum retry delay in seconds (default 10.0)
        non_retryable_statuses: HTTP status codes that should not be retried.
            Defaults to common permanent 4xx codes. Pass frozenset() to retry all.
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful func call, or None if all retries exhausted
    """
    func_name = getattr(func, "__name__", "<unknown>")
    attempts = max(1, int(max_attempts))

    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc, non_retryable_statuses):
                _log_exhaustion(func_name, exc, attempts=attempt)
                return None
            if attempt >= attempts:
                _log_exhaustion(func_name, exc, attempts=attempt)
                return None
            backoff = base_delay * (2 ** (attempt - 1))
            delay_seconds = min(max_delay, backoff)
            _log_retry_attempt(
                func_name,
                exc,
                attempt=attempt,
                delay_seconds=delay_seconds,
            )
    return None
