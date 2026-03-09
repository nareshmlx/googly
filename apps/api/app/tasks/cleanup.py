"""Cleanup tasks for scheduled maintenance."""

from datetime import UTC, datetime, timedelta

import structlog

from app.core.config import settings
from app.core.redis import get_redis
from app.repositories import chat_history as chat_history_repo
from app.repositories import cluster_followup_history as followup_history_repo

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
    cutoff_date = datetime.now(UTC) - timedelta(days=settings.CHAT_HISTORY_RETENTION_DAYS)
    postgres_deleted = 0
    redis_deleted = 0

    logger.info(
        "cleanup.chat_messages.start",
        cutoff_date=cutoff_date.isoformat(),
    )

    # Delete from Postgres
    try:
        postgres_deleted = await chat_history_repo.delete_old_messages_for_service(cutoff_date)
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


async def cleanup_old_cluster_followup_messages(ctx: dict) -> dict:
    """Delete stale cluster follow-up rows older than configured retention."""
    cutoff_date = datetime.now(UTC) - timedelta(days=settings.FOLLOWUP_HISTORY_RETENTION_DAYS)
    deleted = 0
    try:
        deleted = await followup_history_repo.delete_history_older_than_for_service(cutoff_date)
        logger.info(
            "cleanup.cluster_followup.done",
            deleted=deleted,
            cutoff_date=cutoff_date.isoformat(),
        )
    except Exception:
        logger.exception("cleanup.cluster_followup.failed")

    return {
        "deleted": deleted,
        "cutoff_date": cutoff_date.isoformat(),
    }
