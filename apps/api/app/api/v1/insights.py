"""Insights API routes."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.arq import get_arq_pool
from app.core.auth import get_current_user, verify_internal_token
from app.core.config import settings
from app.models.schemas import InsightCard, InsightDetail, InsightRefreshResponse
from app.services import insights as insights_service

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.get("/projects/{project_id}/insights", response_model=list[InsightCard])
async def get_project_insights(
    project_id: UUID,
    current_user: dict = Depends(get_current_user),
):
    """List insight cards for one project."""
    if not settings.INSIGHTS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insights are temporarily disabled",
        )
    try:
        return await insights_service.list_insights(current_user["user_id"], str(project_id))
    except PermissionError:
        raise HTTPException(status_code=404, detail="Project not found")


@router.get("/projects/{project_id}/insights/{insight_id}", response_model=InsightDetail)
async def get_insight_detail(
    project_id: UUID,
    insight_id: UUID,
    current_user: dict = Depends(get_current_user),
):
    """Return detailed insight payload."""
    if not settings.INSIGHTS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insights are temporarily disabled",
        )
    try:
        return await insights_service.get_insight_detail(
            current_user["user_id"],
            str(project_id),
            str(insight_id),
        )
    except PermissionError:
        raise HTTPException(status_code=404, detail="Project not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Insight not found")


@router.get(
    "/projects/{project_id}/insights/{insight_id}/report/stream",
    response_class=StreamingResponse,
)
async def stream_insight_report(
    project_id: UUID,
    insight_id: UUID,
    current_user: dict = Depends(get_current_user),
):
    """Stream report generation for an insight using SSE."""
    if not settings.INSIGHTS_ENABLED or not settings.INSIGHTS_FULL_REPORT_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insight full report is temporarily disabled",
        )

    async def event_generator():
        try:
            async for chunk in insights_service.stream_full_report(
                user_id=current_user["user_id"],
                project_id=str(project_id),
                insight_id=str(insight_id),
            ):
                yield chunk
        except Exception:
            logger.exception(
                "insights.stream_report.error",
                project_id=str(project_id),
                insight_id=str(insight_id),
            )
            yield 'data: {"error": "Unexpected stream error"}\n\n'
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/projects/{project_id}/insights/refresh",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=InsightRefreshResponse,
)
async def refresh_project_insights(
    project_id: UUID,
    current_user: dict = Depends(get_current_user),
    arq_pool=Depends(get_arq_pool),
):
    """Enqueue insight clustering for a project."""
    if not settings.INSIGHTS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insights are temporarily disabled",
        )
    try:
        enqueued = await insights_service.enqueue_cluster(
            current_user["user_id"],
            str(project_id),
            arq_pool,
        )
    except PermissionError:
        raise HTTPException(status_code=404, detail="Project not found")
    except Exception as exc:
        logger.exception("insights.refresh.enqueue_failed", project_id=str(project_id), error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to enqueue insight refresh")

    if enqueued:
        return InsightRefreshResponse(status="accepted", message="Insight refresh enqueued")
    return InsightRefreshResponse(status="skipped", message="Insight refresh already running")
