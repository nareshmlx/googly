"""ARQ task to pre-generate one insight full report in the background."""

from __future__ import annotations

import structlog

from app.services import insights as insights_service

logger = structlog.get_logger(__name__)


async def generate_insight_report(ctx: dict, project_id: str, insight_id: str) -> None:
    """Generate and persist one insight report without requiring a user request."""
    logger.info(
        "generate_insight_report.start",
        project_id=project_id,
        insight_id=insight_id,
    )
    frame_count = 0
    try:
        async for _frame in insights_service.stream_full_report(
            user_id="system",
            project_id=project_id,
            insight_id=insight_id,
            skip_project_ownership_check=True,
            bypass_daily_cap=True,
        ):
            frame_count += 1
    except Exception:
        logger.exception(
            "generate_insight_report.failed",
            project_id=project_id,
            insight_id=insight_id,
        )
        raise

    logger.info(
        "generate_insight_report.done",
        project_id=project_id,
        insight_id=insight_id,
        frame_count=frame_count,
    )
