"""KB router — document upload and status endpoints."""

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.ownership import require_owned_project
from app.core.arq import get_arq_pool
from app.core.auth import get_current_user, verify_internal_token
from app.models.schemas import KBStatusResponse, UploadResponse
from app.services import kb as kb_service

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post("/{project_id}/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    arq_pool=Depends(get_arq_pool),
):
    """Accept a document upload and enqueue async processing."""
    user_id = current_user["user_id"]
    await require_owned_project(project_id, user_id)

    try:
        payload = await kb_service.enqueue_document_upload(
            project_id=project_id,
            user_id=user_id,
            file=file,
            arq_pool=arq_pool,
        )
    except kb_service.KBUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    return UploadResponse(**payload)


@router.get("/{project_id}/status", response_model=KBStatusResponse)
async def get_kb_status(
    project_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return current KB stats for a project — chunk count and last refresh time."""
    user_id = current_user["user_id"]
    project = await require_owned_project(project_id, user_id)
    return KBStatusResponse(**kb_service.build_kb_status(project_id, project))
