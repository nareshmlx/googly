"""KB router — document upload and status endpoints."""

from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.ownership import require_owned_project
from app.core.arq import get_arq_pool
from app.core.auth import get_current_user, verify_internal_token
from app.core.config import settings
from app.core.constants import KBUpload, RedisKeys, RedisTTL
from app.core.redis import get_redis
from app.models.schemas import KBStatusResponse, UploadResponse

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_internal_token)])


async def _read_upload_bytes(file: UploadFile) -> tuple[bytes, int]:
    """Read uploaded bytes with size guard and return (content, size_bytes)."""
    chunks: list[bytes] = []
    total_bytes = 0
    chunk_size = 1024 * 1024

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > settings.KB_MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({total_bytes // (1024 * 1024)} MB). "
                    f"Maximum is {settings.KB_MAX_FILE_SIZE // (1024 * 1024)} MB."
                ),
            )
        chunks.append(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return b"".join(chunks), total_bytes


@router.post("/{project_id}/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    arq_pool=Depends(get_arq_pool),
):
    """
    Accept a document upload and enqueue async processing.

    Returns 202 immediately — the document is not available in the KB
    until the ingest_document worker completes. Poll GET /{project_id}/status
    to check kb_chunk_count.
    """
    user_id = current_user["user_id"]

    # Verify project ownership
    await require_owned_project(project_id, user_id)

    # Validate file type by extension (content_type can be spoofed)
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in KBUpload.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(KBUpload.ALLOWED_EXTENSIONS))}",
        )

    upload_id = str(uuid4())
    content_bytes, size_bytes = await _read_upload_bytes(file)
    staging_key = RedisKeys.UPLOAD_STAGING.format(upload_id=upload_id)

    logger.info(
        "kb.upload.accepted",
        upload_id=upload_id,
        project_id=project_id,
        user_id=user_id,
        filename=filename,
        size_bytes=size_bytes,
    )

    try:
        redis = await get_redis()
        await redis.setex(staging_key, RedisTTL.UPLOAD_STAGING.value, content_bytes)
        await arq_pool.enqueue_job(
            "ingest_document",
            upload_id,
            project_id,
            user_id,
            staging_key,
            filename,
        )
    except Exception:
        try:
            redis = await get_redis()
            await redis.delete(staging_key)
        except Exception:
            logger.warning("kb.upload.staging_cleanup_failed", staging_key=staging_key)
        raise

    return UploadResponse(
        upload_id=upload_id,
        status="processing",
        filename=filename,
        project_id=project_id,
    )


@router.get("/{project_id}/status", response_model=KBStatusResponse)
async def get_kb_status(
    project_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return current KB stats for a project — chunk count and last refresh time."""
    user_id = current_user["user_id"]
    project = await require_owned_project(project_id, user_id)

    return KBStatusResponse(
        project_id=project_id,
        kb_chunk_count=project.get("kb_chunk_count") or 0,
        last_refreshed_at=(
            project["last_refreshed_at"].isoformat() if project.get("last_refreshed_at") else None
        ),
        status="ready" if (project.get("kb_chunk_count") or 0) > 0 else "empty",
    )
