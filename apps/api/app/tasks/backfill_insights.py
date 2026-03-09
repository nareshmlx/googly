"""ARQ task to enqueue initial insight clustering for existing projects."""

from datetime import datetime

import structlog

from app.core.config import settings
from app.core.constants import RedisKeys
from app.core.metrics import insights_backfill_total
from app.core.redis import get_redis
from app.repositories import insights as insights_repo

logger = structlog.get_logger(__name__)


async def _get_queue_lag_snapshot(redis, queue_key: str) -> int | None:
    """Read queue lag using the Redis command that matches the key's actual type."""
    raw_type = await redis.type(queue_key)
    key_type = (
        raw_type.decode("utf-8", errors="ignore")
        if isinstance(raw_type, bytes)
        else str(raw_type or "")
    ).strip().lower()

    if key_type in {"", "none"}:
        return 0
    if key_type == "list":
        return int(await redis.llen(queue_key))
    if key_type == "zset":
        return int(await redis.zcard(queue_key))
    return None


async def backfill_insights(ctx: dict) -> None:
    """Enqueue cluster_project for projects with KB chunks and no insights yet."""
    if not settings.INSIGHTS_ENABLED:
        logger.info("backfill_insights.skipped.disabled")
        return

    redis = await get_redis()
    arq = ctx["redis"]
    discovered = 0
    enqueued = 0
    skipped_locked = 0
    enqueue_failed = 0
    cursor_updated_at: datetime | None = None
    cursor_id: str | None = None
    page_count = 0

    while True:
        page = await insights_repo.list_projects_needing_insight_backfill_page_for_service(
            limit=settings.BACKFILL_INSIGHTS_BATCH_SIZE,
            cursor_updated_at=cursor_updated_at,
            cursor_id=cursor_id,
        )
        if not page:
            break
        page_count += 1
        discovered += len(page)

        for row in page:
            project_id = str(row.get("id") or "").strip()
            if not project_id:
                continue
            enqueue_lock_key = RedisKeys.CLUSTER_ENQUEUE_LOCK.format(project_id=project_id)
            enqueue_slot = await redis.set(
                enqueue_lock_key,
                "1",
                ex=settings.CLUSTER_LOCK_TTL,
                nx=True,
            )
            if not enqueue_slot:
                skipped_locked += 1
                continue
            try:
                await arq.enqueue_job("cluster_project_task", project_id)
                enqueued += 1
            except Exception as exc:
                await redis.delete(enqueue_lock_key)
                enqueue_failed += 1
                logger.warning(
                    "backfill_insights.enqueue_failed",
                    project_id=project_id,
                    error=str(exc),
                )

        last_row = page[-1]
        cursor_updated_at = last_row.get("updated_at")
        cursor_id = str(last_row.get("id") or "").strip() or None

        logger.info(
            "backfill_insights.page_processed",
            page=page_count,
            page_size=len(page),
            cursor_updated_at=cursor_updated_at.isoformat() if cursor_updated_at else None,
            cursor_id=cursor_id,
        )

    insights_backfill_total.labels(status="discovered").inc(discovered)

    if enqueued:
        insights_backfill_total.labels(status="enqueued").inc(enqueued)
    if skipped_locked:
        insights_backfill_total.labels(status="skipped_locked").inc(skipped_locked)
    if enqueue_failed:
        insights_backfill_total.labels(status="enqueue_failed").inc(enqueue_failed)

    queue_lag_snapshot: int | None = None
    try:
        queue_lag_snapshot = await _get_queue_lag_snapshot(redis, RedisKeys.arq_queue())
    except Exception as exc:
        logger.warning("backfill_insights.queue_lag_snapshot_failed", error=str(exc))

    logger.info(
        "backfill_insights.done",
        discovered=discovered,
        page_count=page_count,
        enqueued=enqueued,
        skipped_locked=skipped_locked,
        enqueue_failed=enqueue_failed,
        queue_lag_snapshot=queue_lag_snapshot,
    )
