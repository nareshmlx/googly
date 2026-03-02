import asyncio
from typing import Any
from urllib.parse import urlparse

import asyncpg
import structlog
from sqlalchemy.orm import declarative_base

from app.core.config import settings
from app.core.constants import DatabasePool

logger = structlog.get_logger(__name__)

_pool: asyncpg.Pool | None = None
# Create lock eagerly at module level — asyncio.Lock() is safe to instantiate
# outside a running event loop since Python 3.10 (no longer binds to a loop).
_pool_lock = asyncio.Lock()
Base = declarative_base()


def _parse_db_url(url: str) -> dict[str, Any]:
    """Parse DATABASE_URL into asyncpg connection parameters.

    Strips SQLAlchemy driver prefixes (postgresql+asyncpg://, postgresql+psycopg://)
    so urlparse returns a clean hostname.  asyncpg.create_pool does not accept
    a DSN with a driver suffix.
    """
    import re

    normalised = re.sub(r"^postgresql\+\w+://", "postgresql://", url)
    parsed = urlparse(normalised)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "googly",
        "password": parsed.password or "googly",
        "database": parsed.path.lstrip("/") or "googly",
    }


async def get_db_pool() -> asyncpg.Pool:
    """Return the shared asyncpg connection pool, creating it on first call.

    Double-checked locking with asyncio.Lock prevents the TOCTOU race where
    two concurrent requests both observe _pool is None and each create a pool.
    """
    global _pool

    if _pool is None:
        async with _pool_lock:
            if _pool is None:  # re-check inside the lock
                db_params = _parse_db_url(settings.DATABASE_URL)
                logger.info(
                    "db.pool.create",
                    host=db_params["host"],
                    port=db_params["port"],
                    min_size=DatabasePool.MIN_SIZE,
                    max_size=DatabasePool.MAX_SIZE,
                )
                _pool = await asyncpg.create_pool(
                    host=db_params["host"],
                    port=db_params["port"],
                    user=db_params["user"],
                    password=db_params["password"],
                    database=db_params["database"],
                    min_size=DatabasePool.MIN_SIZE,
                    max_size=DatabasePool.MAX_SIZE,
                )
                logger.info("db.pool.created", pool_size=_pool.get_size())
    return _pool  # type: ignore[return-value]


async def close_db_pools():
    """Close the asyncpg pool on application shutdown."""
    global _pool
    if _pool:
        logger.info("db.pool.closing", pool_size=_pool.get_size())
        await _pool.close()
        _pool = None
