"""Cleanup tasks for scheduled maintenance."""

from datetime import UTC, datetime, timedelta

import structlog

from app.core.db import get_db_pool
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)


async def cleanup_old_chat_messages(ctx: dict) -> dict:
    """
    Delete chat messages older than 90 days from both Postgres and Redis.

    Runs daily at 3am UTC via ARQ cron.
    Returns count of deleted messages for monitoring.

    Why 90 days: Balance between compliance (retain for audit), performance
    (prevent unbounded table growth), and user expectations (long-term history).

    Why both stores: Postgres is the source of truth, but Redis holds working
    sessions. Must clean both to prevent memory leaks.
    """
    cutoff_date = datetime.now(UTC) - timedelta(days=90)
    postgres_deleted = 0
    redis_deleted = 0

    logger.info(
        "cleanup.chat_messages.start",
        cutoff_date=cutoff_date.isoformat(),
    )

    # Delete from Postgres
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM chat_messages
                WHERE created_at < $1
                """,
                cutoff_date,
            )
            # Extract count from result string "DELETE N"
            postgres_deleted = int(result.split()[-1]) if result else 0
            logger.info(
                "cleanup.chat_messages.postgres_complete",
                deleted=postgres_deleted,
            )
    except Exception:
        logger.exception("cleanup.chat_messages.postgres_error")
        # Continue to Redis cleanup even if Postgres fails

    # Delete from Redis (scan for old chat history keys)
    try:
        redis = await get_redis()
        cursor = 0
        while True:
            cursor, keys = await redis.scan(
                cursor,
                match="chat_history:*",
                count=100,
            )
            if keys:
                # Check TTL - if key has no TTL or is expired, delete it
                for key in keys:
                    ttl = await redis.ttl(key)
                    if ttl == -1:  # No expiry set (manual intervention needed)
                        await redis.delete(key)
                        redis_deleted += 1
            if cursor == 0:
                break

        logger.info(
            "cleanup.chat_messages.redis_complete",
            deleted=redis_deleted,
        )
    except Exception:
        logger.exception("cleanup.chat_messages.redis_error")

    logger.info(
        "cleanup.chat_messages.complete",
        postgres_deleted=postgres_deleted,
        redis_deleted=redis_deleted,
        cutoff_date=cutoff_date.isoformat(),
    )

    return {
        "postgres_deleted": postgres_deleted,
        "redis_deleted": redis_deleted,
        "cutoff_date": cutoff_date.isoformat(),
    }
