import asyncio

import redis.asyncio as redis
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

_redis_client: redis.Redis | None = None
# Create lock eagerly at module level — safe since Python 3.10
_redis_lock = asyncio.Lock()


async def get_redis() -> redis.Redis:
    """Return the shared Redis client, creating it on first call.

    Double-checked locking prevents the TOCTOU race under high concurrency.
    """
    global _redis_client

    if _redis_client is None:
        async with _redis_lock:
            if _redis_client is None:
                logger.info(
                    "redis.connecting",
                    url=settings.REDIS_URL.split("@")[-1]
                    if "@" in settings.REDIS_URL
                    else settings.REDIS_URL,
                )
                _redis_client = redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    # Use a larger pool than DatabasePool.MAX_SIZE (50) — the
                    # ingestion worker fans out to 15+ parallel source tools,
                    # each making multiple Redis calls (rate limiter, embed
                    # cache, status writes, circuit breaker).  Under load that
                    # easily exceeds 50 concurrent connections and raises
                    # ConnectionError: Too many connections.
                    # Redis connections are cheap lightweight sockets; 200 is safe.
                    max_connections=200,
                )
                await _redis_client.ping()
                logger.info("redis.connected")
    return _redis_client


async def close_redis():
    """Close the Redis client on application shutdown."""
    global _redis_client
    if _redis_client:
        logger.info("redis.closing")
        await _redis_client.close()
        _redis_client = None
