"""Retry utilities with exponential backoff for external API calls.

Provides simple retry logic with exponential backoff to handle transient
failures from external APIs (network timeouts, 5xx errors, etc.).

Per AGENTS.md Rule 4: Never raises exceptions - returns None on exhaustion.
"""

import asyncio
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


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    non_retryable_statuses: frozenset[int] = _DEFAULT_NON_RETRYABLE_STATUSES,
    **kwargs: Any,
) -> T | None:
    """Retry async function with exponential backoff on failure.

    Retries on transient exceptions (network errors, 5xx), with delay doubling
    after each attempt. Returns None if all retries exhausted (never raises).

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
    attempt = 0
    while attempt < max_attempts:
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            attempt += 1
            status_code = e.response.status_code
            if status_code in non_retryable_statuses:
                # Permanent failure — do not retry, do not sleep
                logger.warning(
                    "retry.non_retryable_http_error",
                    func=func.__name__,
                    status_code=status_code,
                    error=str(e),
                )
                return None

            if attempt >= max_attempts:
                logger.error(
                    "retry.exhausted",
                    func=func.__name__,
                    attempts=attempt,
                    status_code=status_code,
                    error=str(e),
                )
                return None

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "retry.attempt",
                func=func.__name__,
                attempt=attempt,
                max_attempts=max_attempts,
                delay_seconds=delay,
                status_code=status_code,
                error=str(e),
            )
            await asyncio.sleep(delay)
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                logger.error(
                    "retry.exhausted",
                    func=func.__name__,
                    attempts=attempt,
                    error=str(e),
                )
                # Per Rule 4: return None instead of raising
                return None

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "retry.attempt",
                func=func.__name__,
                attempt=attempt,
                max_attempts=max_attempts,
                delay_seconds=delay,
                error=str(e),
            )
            await asyncio.sleep(delay)

    # Should never reach here, but type checker needs it
    return None
