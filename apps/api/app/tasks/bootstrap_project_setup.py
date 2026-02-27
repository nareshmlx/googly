"""ARQ bootstrap task for async project setup after create/upload flows."""

import json
from datetime import UTC, datetime
from hashlib import sha256

import structlog

from app.core.arq import get_arq_pool
from app.core.db import get_db_pool
from app.kb.embedder import embed_texts
from app.kb.intent_extractor import extract_document_intent, merge_intents
from app.repositories import project as project_repo
from app.services import project as project_service

logger = structlog.get_logger(__name__)


def _upload_signature(upload_ids: list[str]) -> str:
    """Compute stable upload signature for status/idempotency traces."""
    if not upload_ids:
        return ""
    return sha256("|".join(upload_ids).encode("utf-8")).hexdigest()


async def _set_status(
    project_id: str,
    *,
    status: str,
    phase: str,
    progress_percent: int,
    message: str | None = None,
    error: str | None = None,
    upload_ids: list[str] | None = None,
    upload_signature: str = "",
    job_id: str | None = None,
) -> dict:
    """Proxy setup status updates via service helper."""
    return await project_service.set_project_setup_status(
        project_id,
        status=status,
        phase=phase,
        progress_percent=progress_percent,
        message=message,
        error=error,
        upload_ids=upload_ids,
        upload_signature=upload_signature,
        job_id=job_id,
    )


async def _ingest_is_active(project_id: str) -> bool:
    """Check whether project ingest lifecycle is already queued/running."""
    status = await project_service.get_project_ingest_status(project_id)
    return str(status.get("status") or "").lower() in {"queued", "running"}


def _coerce_intent(value: object) -> dict:
    """Normalize persisted structured_intent payload into a mapping."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def bootstrap_project_setup(
    ctx: dict,
    project_id: str,
    user_id: str,
    upload_ids: list[str] | None = None,
) -> None:
    """Run async setup phases and refresh source ingestion after intent merge."""
    canonical_upload_ids = sorted({str(value).strip() for value in (upload_ids or []) if str(value).strip()})
    upload_signature = _upload_signature(canonical_upload_ids)

    await _set_status(
        project_id,
        status="running",
        phase="processing_documents",
        progress_percent=20,
        message="Processing uploaded documents.",
        upload_ids=canonical_upload_ids,
        upload_signature=upload_signature,
    )

    pool = await get_db_pool()
    project = await project_repo.fetch_project(pool, project_id, user_id)
    if not project:
        await _set_status(
            project_id,
            status="degraded",
            phase="processing_documents",
            progress_percent=100,
            message="Project not found for bootstrap setup.",
            error="project_not_found",
            upload_ids=canonical_upload_ids,
            upload_signature=upload_signature,
        )
        return

    try:
        summaries = await project_repo.fetch_upload_chunk_summaries(
            pool,
            project_id,
            canonical_upload_ids,
            limit=40,
        )

        await _set_status(
            project_id,
            status="running",
            phase="refining_intent",
            progress_percent=45,
            message="Refining intent from document evidence.",
            upload_ids=canonical_upload_ids,
            upload_signature=upload_signature,
        )

        base_intent = _coerce_intent(project.get("structured_intent"))
        doc_intent = await extract_document_intent(summaries) if summaries else {}

        await _set_status(
            project_id,
            status="running",
            phase="merging_intent",
            progress_percent=65,
            message="Merging base and document-derived intent.",
            upload_ids=canonical_upload_ids,
            upload_signature=upload_signature,
        )

        merged_intent = merge_intents(base_intent, doc_intent)
        merged_intent["intent_sources"] = {
            "description": base_intent,
            "documents": doc_intent,
        }
        merged_intent["intent_merge_version"] = "v1"
        if summaries:
            merged_intent["last_doc_refined_at"] = datetime.now(UTC).isoformat()

        await project_repo.update_project_intent(pool, project_id, merged_intent)

        embedding_input = f"{project.get('description', '')}\n{json.dumps(merged_intent, sort_keys=True)}"
        vectors = await embed_texts([embedding_input])
        if vectors:
            await project_repo.update_project_intent_embedding(pool, project_id, vectors[0])

        await _set_status(
            project_id,
            status="running",
            phase="refreshing_sources",
            progress_percent=85,
            message="Refreshing external sources.",
            upload_ids=canonical_upload_ids,
            upload_signature=upload_signature,
        )

        if not await _ingest_is_active(project_id):
            arq_pool = await get_arq_pool()
            job = await arq_pool.enqueue_job("ingest_project", project_id)
            now = datetime.now(UTC).isoformat()
            await project_service.set_project_ingest_status(
                project_id,
                status="queued",
                message="Ingestion queued from bootstrap setup.",
                queued_at=now,
                updated_at=now,
                job_id=str(getattr(job, "job_id", "") or ""),
            )

        await _set_status(
            project_id,
            status="ready",
            phase="ready",
            progress_percent=100,
            message="Project setup completed.",
            upload_ids=canonical_upload_ids,
            upload_signature=upload_signature,
        )
    except Exception as exc:
        logger.exception("bootstrap_project_setup.failed", project_id=project_id)
        await _set_status(
            project_id,
            status="degraded",
            phase="refreshing_sources",
            progress_percent=100,
            message="Project setup completed with issues.",
            error=str(exc),
            upload_ids=canonical_upload_ids,
            upload_signature=upload_signature,
        )
