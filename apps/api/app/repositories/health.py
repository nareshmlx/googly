"""Health repository probes for dependency readiness checks."""

from __future__ import annotations

import structlog

from app.core.db import get_db_pool

logger = structlog.get_logger(__name__)


async def check_db_ready() -> bool:
    """Return True when a basic DB probe succeeds."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        logger.exception("health.db_probe_failed")
        return False
