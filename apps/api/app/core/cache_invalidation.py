"""Shared project cache invalidation logic.

Extracted from ingest_document.py and ingest_project.py to eliminate duplication.
Both task modules import from here — single source of truth.
"""

import structlog

from app.core.cache_version import bump_project_cache_version
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)


async def invalidate_project_caches(project_id: str) -> None:
    """Invalidate all caches for a project after KB update.

    Bumps project cache version so semantic + KB hot cache keys rotate immediately,
    then clears project-scoped search cache keys via SCAN.

    Cache invalidation failure is logged but does not crash the ingestion
    task — data consistency is maintained even if caches remain stale.
    """
    try:
        redis = await get_redis()
        new_version = await bump_project_cache_version(redis, project_id)
        patterns = [
            f"search:cache:{project_id}:*",
        ]

        deleted_count = 0
        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis.delete(*keys)
                    deleted_count += len(keys)
                if cursor == 0:
                    break

        logger.info(
            "cache_invalidation.complete",
            project_id=project_id,
            cache_version=new_version,
            keys_deleted=deleted_count,
        )
    except Exception:
        logger.exception(
            "cache_invalidation.failed",
            project_id=project_id,
        )
