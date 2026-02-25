"""refresh_project ARQ task — re-pulls from Instagram for daily/weekly projects.

Uses oldest_timestamp = last_refreshed_at so only new content since the last
run is fetched. Avoids re-ingesting already-stored content.
"""

import structlog

from app.core.db import get_db_pool
from app.tasks.ingest_project import _run_ingestion

logger = structlog.get_logger(__name__)


async def refresh_project(
    ctx: dict,
    project_id: str,
    oldest_timestamp: int | None = None,
) -> None:
    """
    Re-pull Instagram content for a project, fetching only content newer
    than last_refreshed_at to avoid duplicate ingestion.

    Delegates to ingest_project._run_ingestion with oldest_timestamp set.
    """
    if oldest_timestamp is None:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT last_refreshed_at FROM projects WHERE id = $1::uuid",
                project_id,
            )

        if not row:
            logger.warning("refresh_project.not_found", project_id=project_id)
            return

        last_refreshed = row["last_refreshed_at"]
        oldest_timestamp = int(last_refreshed.timestamp()) if last_refreshed else None

    logger.info(
        "refresh_project.start",
        project_id=project_id,
        oldest_timestamp=oldest_timestamp,
    )
    await _run_ingestion(ctx, project_id, oldest_timestamp=oldest_timestamp)


async def refresh_due_projects(ctx: dict) -> None:
    """
    Cron entry point — enqueues refresh_project for every project that is
    overdue for its scheduled refresh strategy.

    Runs every 6 hours. Projects with refresh_strategy='once' or 'on_demand'
    are never touched by this cron.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, last_refreshed_at FROM projects
            WHERE
                (refresh_strategy = 'daily'
                 AND (last_refreshed_at IS NULL OR last_refreshed_at < NOW() - INTERVAL '1 day'))
            OR
                (refresh_strategy = 'weekly'
                 AND (last_refreshed_at IS NULL OR last_refreshed_at < NOW() - INTERVAL '7 days'))
            """
        )

    projects = [
        {
            "id": r["id"],
            "oldest_timestamp": int(r["last_refreshed_at"].timestamp())
            if r["last_refreshed_at"]
            else None,
        }
        for r in rows
    ]
    project_ids = [p["id"] for p in projects]
    logger.info("refresh_due_projects.found", count=len(project_ids))

    arq = ctx["redis"]
    enqueued_count = 0
    failed_count = 0
    for project in projects:
        pid = project["id"]
        try:
            await arq.enqueue_job("refresh_project", pid, project["oldest_timestamp"])
            enqueued_count += 1
            logger.info("refresh_due_projects.enqueued", project_id=pid)
        except Exception as exc:
            failed_count += 1
            logger.warning(
                "refresh_due_projects.enqueue_failed",
                project_id=pid,
                error=str(exc),
            )

    logger.info(
        "refresh_due_projects.summary",
        found_count=len(project_ids),
        enqueued_count=enqueued_count,
        failed_count=failed_count,
    )
