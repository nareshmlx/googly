"""ARQ pool — lazily initialised, injected via FastAPI Depends.

Kept separate from redis.py because ARQ uses its own connection type (ArqRedis)
which has different methods (enqueue_job) vs the general Redis client used for
caching (get, setex, pipeline).
"""

import asyncio

import structlog
from arq import create_pool
from arq.connections import RedisSettings

from app.core.config import settings

logger = structlog.get_logger(__name__)

_arq_pool = None
_arq_lock: asyncio.Lock | None = None


async def get_arq_pool():
    """Return the shared ARQ pool, creating it on first call.

    Double-checked locking prevents the TOCTOU race where concurrent requests
    each try to create a separate ARQ connection pool.
    """
    global _arq_pool, _arq_lock
    if _arq_lock is None:
        _arq_lock = asyncio.Lock()

    if _arq_pool is None:
        async with _arq_lock:
            if _arq_pool is None:
                _arq_pool = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
                logger.info("arq.pool.created")
    return _arq_pool


async def close_arq_pool():
    """Close the ARQ pool on app shutdown."""
    global _arq_pool
    if _arq_pool:
        await _arq_pool.close()
        _arq_pool = None
        logger.info("arq.pool.closed")
