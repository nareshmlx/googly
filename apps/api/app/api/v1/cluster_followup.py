"""Cluster follow-up API routes."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.auth import get_current_user, verify_internal_token
from app.core.config import settings
from app.core.user_rate_limiter import check_user_rate_limit
from app.models.schemas import FollowupMessage, FollowupRequest
from app.services import cluster_followup as followup_service

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post("/insights/{insight_id}/followup", response_class=StreamingResponse)
async def stream_followup(
    insight_id: UUID,
    request: FollowupRequest,
    current_user: dict = Depends(get_current_user),
):
    """Stream strict source-scoped follow-up answer for one insight."""
    if not settings.INSIGHTS_ENABLED or not settings.INSIGHTS_FOLLOWUP_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insight follow-up is temporarily disabled",
        )

    await check_user_rate_limit(current_user["user_id"], "followup")

    async def event_generator():
        try:
            async for chunk in followup_service.stream_cluster_followup(
                user_id=current_user["user_id"],
                insight_id=str(insight_id),
                message=request.message,
            ):
                yield chunk
        except Exception:
            logger.exception("cluster_followup.stream.error", insight_id=str(insight_id))
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


@router.get("/insights/{insight_id}/followup/history", response_model=list[FollowupMessage])
async def get_followup_history(
    insight_id: UUID,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """Return persisted follow-up history for an insight and current user (paginated)."""
    if not settings.INSIGHTS_ENABLED or not settings.INSIGHTS_FOLLOWUP_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insight follow-up is temporarily disabled",
        )
    try:
        return await followup_service.get_followup_history(
            current_user["user_id"],
            str(insight_id),
            limit=limit,
            offset=offset,
        )
    except (PermissionError, LookupError):
        raise HTTPException(status_code=404, detail="Insight not found")

