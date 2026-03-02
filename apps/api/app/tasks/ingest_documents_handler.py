"""Document handling logic (fulltext enrichment, lifecycle status)."""

import json
from datetime import UTC, datetime

import structlog

from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.metrics import fulltext_resolve_total
from app.kb.ingester import RawDocument
from app.repositories import source_asset as source_asset_repo
from app.services.fulltext_resolver import resolve_fulltext_url
from app.tasks.ingest_filters import _filter_relevance
from app.tasks.ingest_utils import SOCIAL_SOURCES, _is_document_new_enough

logger = structlog.get_logger(__name__)


async def _prepare_kept_documents_for_source(
    *,
    result,
    source: str,
    source_enabled: bool,
    oldest_timestamp: int | None,
    strict_social_terms: list[str],
    intent_text: str,
    redis,
    expansion_meta: dict[str, dict[str, int | bool | str]],
    intent_embedding: list[float] | None = None,
) -> tuple[list[RawDocument], dict[str, int | str | bool], bool]:
    """Normalize and relevance-filter one source result batch."""
    if not source_enabled:
        return [], {"fetched": 0, "kept": 0, "filtered_out": 0, "reason": "source_disabled"}, False

    if isinstance(result, BaseException):
        diagnostics: dict[str, int | str | bool] = {
            "fetched": 0,
            "kept": 0,
            "filtered_out": 0,
            "reason": "source_failed",
        }
        diagnostics.update(expansion_meta.get(source, {}))
        return [], diagnostics, False

    if not result:
        diagnostics = {
            "fetched": 0,
            "kept": 0,
            "filtered_out": 0,
            "reason": "no_results_from_source",
        }
        diagnostics.update(expansion_meta.get(source, {}))
        return [], diagnostics, True

    filtered_result = list(result)
    if oldest_timestamp is not None:
        filtered_result = [
            doc for doc in filtered_result if _is_document_new_enough(doc, oldest_timestamp)
        ]
        if not filtered_result:
            diagnostics = {
                "fetched": 0,
                "kept": 0,
                "filtered_out": 0,
                "reason": "all_older_than_refresh_watermark",
            }
            diagnostics.update(expansion_meta.get(source, {}))
            return [], diagnostics, True

    filter_items: list[dict] = []
    for index, doc in enumerate(filtered_result):
        metadata = doc.metadata or {}
        filter_items.append(
            {
                "_idx": index,
                "title": doc.title or "",
                "abstract": doc.content,
                "content": doc.content,
                "claims": metadata.get("claims") or metadata.get("claims_snippet") or "",
                "source_id": doc.source_id or "",
                "url": metadata.get("url") or "",
                "_fallback_origin": bool(metadata.get("fallback_origin"))
                or "fallback" in str(metadata.get("tool") or "").lower(),
            }
        )

    fetched_count = len(filtered_result)
    social_terms_for_source = strict_social_terms if source in SOCIAL_SOURCES else []
    filtered_items = await _filter_relevance(
        filter_items,
        intent_text,
        source,
        redis,
        must_match_terms=social_terms_for_source if source in SOCIAL_SOURCES else [],
        social_match_terms=social_terms_for_source,
        intent_embedding=intent_embedding,
    )
    kept_indexes = {
        int(item.get("_idx", -1))
        for item in filtered_items
        if isinstance(item, dict) and isinstance(item.get("_idx"), int)
    }
    kept_docs = [doc for index, doc in enumerate(filtered_result) if index in kept_indexes]
    kept_count = len(kept_docs)
    diagnostics = {
        "fetched": fetched_count,
        "kept": kept_count,
        "filtered_out": max(0, fetched_count - kept_count),
        "reason": (
            "strict_relevance_filtered"
            if source in SOCIAL_SOURCES and kept_count == 0 and fetched_count > 0
            else "ok"
        ),
    }
    diagnostics.update(expansion_meta.get(source, {}))
    return kept_docs, diagnostics, True


async def _set_ingest_status(
    redis,
    project_id: str,
    *,
    status: str,
    message: str | None = None,
    queued_at: str | None = None,
    started_at: str | None = None,
    updated_at: str | None = None,
    finished_at: str | None = None,
    source_counts: dict | None = None,
    source_diagnostics: dict | None = None,
    fulltext_enqueued: int = 0,
    total_chunks: int | None = None,
    job_id: str | None = None,
) -> None:
    """Persist ingest lifecycle status for API/UI visibility.

    Uses a single JSON blob in Redis (string type) to match ProjectService reader.
    """
    if redis is None:
        return
    key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)

    # Win #10: Correct Redis key usage + JSON serialization
    # Initialize with required fields
    data: dict = {
        "project_id": project_id,
        "status": status,
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
        "source_counts": source_counts or {},
        "source_diagnostics": source_diagnostics or {},
        "fulltext_enqueued": int(fulltext_enqueued),
    }

    # Optional fields
    if message is not None:
        data["message"] = message
    if queued_at is not None:
        data["queued_at"] = queued_at
    if started_at is not None:
        data["started_at"] = started_at
    if finished_at is not None:
        data["finished_at"] = finished_at
    if total_chunks is not None:
        data["total_chunks"] = int(total_chunks)
    if job_id is not None:
        data["job_id"] = str(job_id)

    try:
        # Use setex to match ProjectService.get() which expects a string/blob
        await redis.setex(key, RedisTTL.PROJECT_INGEST_STATUS.value, json.dumps(data))
    except Exception:
        logger.exception("set_ingest_status.failed", project_id=project_id)


async def _schedule_fulltext_enrichment(ctx: dict, pool, documents: list[RawDocument]) -> int:
    """Resolve/upsert fulltext assets and enqueue enrichment tasks for eligible docs."""
    if not settings.ENABLE_FULLTEXT_ENRICHMENT:
        return 0
    redis = ctx.get("redis")
    if redis is None:
        logger.warning("fulltext_enrichment.redis_missing")
        return 0

    scheduled = 0
    for doc in documents:
        if doc.source not in {"paper", "patent"}:
            continue
        metadata = dict(doc.metadata or {})
        if str(metadata.get("content_level") or "abstract") == "fulltext":
            continue

        resolved = resolve_fulltext_url(doc)
        fulltext_resolve_total.labels(source=doc.source, status=resolved.status).inc()
        if resolved.status != "success" or not resolved.resolved_url or not resolved.canonical_url:
            continue

        source_url = str(
            metadata.get("open_access_url")
            or metadata.get("pdf_url")
            or metadata.get("url")
            or resolved.resolved_url
        )
        asset_id = await source_asset_repo.upsert_source_asset(
            pool,
            project_id=doc.project_id,
            user_id=doc.user_id,
            source=doc.source,
            source_id=str(doc.source_id or ""),
            title=str(doc.title or ""),
            source_url=source_url,
            resolved_url=resolved.resolved_url,
            canonical_url=resolved.canonical_url,
            source_fetcher=resolved.source_fetcher,
        )
        if not asset_id:
            continue
        # Using redis.enqueue_job assuming redis is an Arq container or similar wrapper as used in codebase
        try:
            await redis.enqueue_job("ingest_source_asset", asset_id, _job_id=f"fulltext:{asset_id}")
            scheduled += 1
        except Exception:
            logger.exception("fulltext_enrichment.enqueue_failed", asset_id=asset_id)

    return scheduled
