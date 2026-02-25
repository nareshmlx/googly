import asyncio

import redis.asyncio as redis
import structlog

from app.core.config import settings
from app.core.constants import DatabasePool

logger = structlog.get_logger(__name__)

_redis_client: redis.Redis | None = None
_redis_lock: asyncio.Lock | None = None


async def get_redis() -> redis.Redis:
    """Return the shared Redis client, creating it on first call.

    Double-checked locking prevents the TOCTOU race under high concurrency.
    """
    global _redis_client, _redis_lock
    if _redis_lock is None:
        _redis_lock = asyncio.Lock()

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
                    max_connections=DatabasePool.MAX_SIZE,
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
