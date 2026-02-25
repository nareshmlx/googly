import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.ownership import require_owned_project
from app.core.auth import get_current_user, verify_internal_token
from app.core.db import get_db_pool
from app.core.user_rate_limiter import check_chat_rate_limit
from app.models.schemas import ChatRequest
from app.repositories import chat_history as chat_history_repo
from app.services.chat import get_chat_history_messages, stream_response

logger = structlog.get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post("/", response_class=StreamingResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Stream chat response via SSE.

    Verifies project ownership before streaming — a user may not query a
    project they do not own, even if they know its UUID.

    Rate limited to 100 requests per 60 seconds per user.
    """
    # Check rate limit
    await check_chat_rate_limit(current_user)

    user_id = current_user["user_id"]

    # Ownership check — project_id is already UUID-validated by ChatRequest
    await require_owned_project(request.project_id, user_id)

    session_id = request.session_id or f"session_{user_id}_{request.project_id}"

    # Session ownership check — if user provided a custom session_id, verify they own it
    if request.session_id:
        pool = await get_db_pool()
        session_owned = await chat_history_repo.verify_session_ownership(
            pool, request.project_id, user_id, request.session_id
        )
        if not session_owned:
            raise HTTPException(
                status_code=403,
                detail="Session does not belong to current user",
            )

    logger.info(
        "chat.request",
        user_id=user_id,
        project_id=request.project_id,
        session_id=session_id,
        query_preview=request.query[:80],
    )

    async def event_generator():
        async for chunk in stream_response(
            query=request.query,
            project_id=request.project_id,
            user_id=user_id,
            session_id=session_id,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{project_id}")
async def get_chat_history(
    project_id: str,
    session_id: str = Query(
        ...,
        description="Session identifier (required query parameter)",
        examples=["session_user123_proj456"],
    ),
    current_user: dict = Depends(get_current_user),
):
    """
    Return the chat history for a given project and session as an ordered list
    of {role, content} messages.

    **Required parameters:**
    - `project_id` (path): UUID of the project
    - `session_id` (query): Session identifier (e.g., `?session_id=session_xyz`)

    History is loaded from durable Postgres storage (source of truth), with
    Redis fallback handled in the service for legacy pre-migration sessions.

    **Returns 403** if the session does not belong to the current user.
    **Returns 404** if the project is not found or not owned by the user.
    """
    user_id = current_user["user_id"]

    await require_owned_project(project_id, user_id)
    pool = await get_db_pool()

    # Session ownership check — verify session belongs to current user
    session_owned = await chat_history_repo.verify_session_ownership(
        pool, project_id, user_id, session_id
    )
    if not session_owned:
        raise HTTPException(
            status_code=403,
            detail="Session does not belong to current user",
        )

    return await get_chat_history_messages(
        project_id=project_id,
        user_id=user_id,
        session_id=session_id,
    )
