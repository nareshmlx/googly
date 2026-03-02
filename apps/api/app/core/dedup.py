"""Request deduplication to prevent thundering herd on external APIs.

When 1000 users search for "retinol" simultaneously:
1. Process the first request normally
2. For the next 999 requests, wait for the first one to complete
3. Return the same cached result to all 1000 users

This prevents API rate limit exhaustion and improves performance.

Integrated with AGENTS.md Rule 7 (designed for 10k concurrent users).
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


async def deduplicate_request(key: str, fn: Callable[[], Awaitable[T]]) -> T:
    """
    Deduplicate concurrent identical requests across all worker pods globally.

    Uses a Redis lock to elect a primary worker for a given request.
    Secondary workers poll Redis for the result.
    """
    import json

    from app.core.redis import get_redis

    redis = await get_redis()
    lock_key = f"sys:dedup:lock:{key}"
    result_key = f"sys:dedup:res:{key}"

    # Try to become the primary worker
    acquired = await redis.set(lock_key, "1", nx=True, ex=120)
    if acquired:
        logger.debug("dedup.first_request", key=key[:50])
        try:
            result = await fn()
            # Serialize and broadcast result globally
            await redis.setex(result_key, 60, json.dumps({"status": "ok", "data": result}))
            logger.debug("dedup.completed", key=key[:50])
            return result
        except Exception as exc:
            await redis.setex(result_key, 60, json.dumps({"status": "error", "message": str(exc)}))
            logger.warning("dedup.error", key=key[:50], error=str(exc))
            raise
        finally:
            await redis.delete(lock_key)
    else:
        # We are a secondary worker, poll for globally resolved result
        import asyncio

        logger.debug("dedup.waiting_globally", key=key[:50])
        for _ in range(600):  # Wait up to 60 seconds
            raw = await redis.get(result_key)
            if raw:
                try:
                    payload = json.loads(raw)
                    if payload.get("status") == "error":
                        raise Exception(payload.get("message", "Unknown error in primary worker"))
                    logger.debug("dedup.reused", key=key[:50])
                    return payload.get("data")  # type: ignore
                except json.JSONDecodeError:
                    pass
            await asyncio.sleep(0.1)

        # Fallback if primary process died before writing result
        logger.warning("dedup.timeout_fallback", key=key[:50])
        return await fn()
