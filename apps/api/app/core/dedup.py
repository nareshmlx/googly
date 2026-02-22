"""Request deduplication to prevent thundering herd on external APIs.

When 1000 users search for "retinol" simultaneously:
1. Process the first request normally
2. For the next 999 requests, wait for the first one to complete
3. Return the same cached result to all 1000 users

This prevents API rate limit exhaustion and improves performance.

Integrated with AGENTS.md Rule 7 (designed for 10k concurrent users).
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Global dict: query_hash -> asyncio.Future
# This is safe because asyncio is single-threaded (no race conditions)
_inflight_requests: dict[str, asyncio.Future[Any]] = {}
_inflight_lock = asyncio.Lock()


async def deduplicate_request(
    key: str,
    fn: Callable[[], Awaitable[T]],
) -> T:
    """
    Deduplicate concurrent identical requests.

    If multiple callers request the same key simultaneously:
    - First caller executes fn()
    - Other callers wait for the first one to complete
    - All callers receive the same result

    Args:
        key: Unique identifier for this request (e.g., cache key)
        fn: Async function to call if this is the first request

    Returns:
        Result of fn()

    Raises:
        Exception: If fn() raises an exception, all waiting callers receive it

    Example:
        result = await deduplicate_request(
            "papers:semantic_scholar:abc123",
            lambda: search_semantic_scholar_impl("machine learning", "papers:semantic_scholar:abc123")
        )
    """
    # Check if this request is already in flight
    async with _inflight_lock:
        if key in _inflight_requests:
            # Another request for this key is already in flight
            logger.debug("dedup.waiting", key=key[:50])
            future = _inflight_requests[key]
            is_first_request = False
        else:
            # This is the first request for this key
            logger.debug("dedup.first_request", key=key[:50])
            future = asyncio.Future()
            _inflight_requests[key] = future
            is_first_request = True

    # If we're the first request, execute the function
    if is_first_request:
        try:
            result = await fn()
            future.set_result(result)
            logger.debug("dedup.completed", key=key[:50])
        except Exception as exc:
            future.set_exception(exc)
            logger.warning("dedup.error", key=key[:50], error=str(exc))
            raise
        finally:
            # Clean up the future from inflight dict
            async with _inflight_lock:
                _inflight_requests.pop(key, None)
    else:
        # We're a waiter, just await the future
        logger.debug("dedup.reused", key=key[:50])

    return await future
