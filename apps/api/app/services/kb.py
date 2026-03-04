"""KB service — upload orchestration and status shaping."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TypedDict
from uuid import uuid4

import structlog
from arq import ArqRedis
from fastapi import UploadFile

from app.core.config import settings
from app.core.constants import KBUpload, RedisKeys, RedisTTL
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)


class KBStatusPayload(TypedDict):
    """Typed payload shape for KB status responses."""

    project_id: str
    kb_chunk_count: int
    last_refreshed_at: str | None
    status: str


@dataclass(frozen=True)
class KBUploadError(Exception):
    """Domain error for KB upload validation/processing failures."""

    status_code: int
    detail: str


async def _read_upload_bytes(file: UploadFile) -> tuple[bytearray, int]:
    """Read uploaded bytes with size guard and return (content, size_bytes)."""
    buffer = bytearray()
    total_bytes = 0
    chunk_size = 1024 * 1024

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > settings.KB_MAX_FILE_SIZE:
            raise KBUploadError(
                status_code=413,
                detail=(
                    f"File too large ({total_bytes // (1024 * 1024)} MB). "
                    f"Maximum is {settings.KB_MAX_FILE_SIZE // (1024 * 1024)} MB."
                ),
            )
        buffer.extend(chunk)

    if total_bytes == 0:
        raise KBUploadError(status_code=400, detail="Uploaded file is empty")

    return buffer, total_bytes


async def enqueue_document_upload(
    *,
    project_id: str,
    user_id: str,
    file: UploadFile,
    arq_pool: ArqRedis,
) -> dict[str, str]:
    """Validate/stage upload content and enqueue async KB ingest task."""
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in KBUpload.ALLOWED_EXTENSIONS:
        raise KBUploadError(
            status_code=400,
            detail=(
                f"Unsupported file type '.{ext}'. "
                f"Allowed: {', '.join(sorted(KBUpload.ALLOWED_EXTENSIONS))}"
            ),
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
        encoded = base64.b64encode(content_bytes).decode("ascii")
        await redis.setex(staging_key, RedisTTL.UPLOAD_STAGING.value, encoded)
        await arq_pool.enqueue_job(
            "ingest_document",
            upload_id,
            project_id,
            user_id,
            staging_key,
            filename,
        )
    except Exception:
        logger.exception(
            "kb.upload.enqueue_failed",
            upload_id=upload_id,
            project_id=project_id,
            user_id=user_id,
            filename=filename,
            staging_key=staging_key,
        )
        try:
            redis = await get_redis()
            await redis.delete(staging_key)
        except Exception:
            logger.warning("kb.upload.staging_cleanup_failed", staging_key=staging_key)
        raise

    return {
        "upload_id": upload_id,
        "status": "processing",
        "filename": filename,
        "project_id": project_id,
    }


def build_kb_status(project_id: str, project: dict) -> KBStatusPayload:
    """Build API-ready KB status payload from project row."""
    kb_chunk_count = project.get("kb_chunk_count") or 0
    return {
        "project_id": project_id,
        "kb_chunk_count": kb_chunk_count,
        "last_refreshed_at": (
            project["last_refreshed_at"].isoformat() if project.get("last_refreshed_at") else None
        ),
        "status": "ready" if kb_chunk_count > 0 else "empty",
    }
