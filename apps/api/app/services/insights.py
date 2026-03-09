"""Insights service orchestration for list/detail/report streaming and clustering enqueue."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from time import perf_counter

import structlog
from arq import ArqRedis

from app.agents.insight_generator import (
    generate_full_report,
    link_evidence_references,
    refine_full_report,
)
from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.metrics import (
    insights_full_report_duration_seconds,
    insights_full_report_total,
)
from app.core.redis import get_redis
from app.core.serialization import sse_frame as _sse_frame
from app.core.serialization import to_dict as _to_dict
from app.core.serialization import to_list as _to_list
from app.kb.ingester import _select_summary_excerpts, _summarize_documents
from app.models.schemas import InsightCard, InsightDetail
from app.repositories import insights as insights_repo
from app.repositories import knowledge as knowledge_repo
from app.services import project as project_service

logger = structlog.get_logger(__name__)


def _cluster_enqueue_ttl_seconds() -> int:
    """Keep cluster enqueue dedupe alive for the full worker timeout window."""
    return max(int(settings.CLUSTER_LOCK_TTL), int(settings.ARQ_WORKER_JOB_TIMEOUT))


def _merge_chunk_rows(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Merge chunk row lists in stable order, deduplicated by chunk id."""
    merged: list[dict] = []
    seen_ids: set[str] = set()
    for row in primary + secondary:
        chunk_id = str(row.get("id") or "").strip()
        if not chunk_id or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        merged.append(row)
    return merged


def _missing_source_summary_doc_ids(source_docs: list[dict]) -> list[str]:
    """Return source-doc IDs that still lack a generated summary."""
    missing_ids: list[str] = []
    seen_ids: set[str] = set()
    for row in source_docs or []:
        doc_id = str((row or {}).get("id") or "").strip()
        summary = str((row or {}).get("summary") or "").strip()
        summary_origin = str((row or {}).get("summary_origin") or "").strip().lower()
        has_generated_summary = bool(summary) and summary_origin not in {"fallback", "missing"}
        if not doc_id or has_generated_summary or doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        missing_ids.append(doc_id)
    return missing_ids


async def _ensure_project_owned(user_id: str, project_id: str) -> None:
    """Raise permission error when project is missing or not owned by the user."""
    project = await project_service.get_project(project_id, user_id)
    if not project:
        raise PermissionError("Project not found")


async def ensure_source_doc_summaries(project_id: str, source_doc_ids: list[str]) -> None:
    """Ensure source documents have generated summaries before insight rendering."""
    if not settings.DOC_SUMMARIZE_ENABLED:
        return
    doc_ids = [str(doc_id).strip() for doc_id in source_doc_ids if str(doc_id).strip()]
    if not doc_ids:
        return

    try:
        pending_doc_ids = await knowledge_repo.list_documents_without_summary_for_service(doc_ids=doc_ids)
        if not pending_doc_ids:
            return

        chunk_rows = await insights_repo.get_chunks_for_documents_for_service(
            project_id,
            list(pending_doc_ids),
            limit=400,
        )
        if not chunk_rows:
            return

        chunks_by_doc: dict[str, list[str]] = {}
        for row in chunk_rows:
            doc_id = str(row.get("document_id") or "").strip()
            if not doc_id or doc_id not in pending_doc_ids:
                continue
            chunks_by_doc.setdefault(doc_id, []).append(str(row.get("content") or ""))

        selected_chunks_by_doc: dict[str, list[str]] = {}
        for doc_id, chunk_values in chunks_by_doc.items():
            excerpts = _select_summary_excerpts(chunk_values, max_excerpts=3)
            if excerpts:
                selected_chunks_by_doc[doc_id] = excerpts
        if not selected_chunks_by_doc:
            return

        summaries = await _summarize_documents(selected_chunks_by_doc)
        filtered = {
            doc_id: summary
            for doc_id, summary in summaries.items()
            if summary and doc_id in pending_doc_ids
        }
        if not filtered:
            return
        updated = await knowledge_repo.update_document_summaries_for_service(summaries=filtered)
        logger.info(
            "insights.source_summary.backfill_done",
            project_id=project_id,
            requested=len(selected_chunks_by_doc),
            updated=updated,
        )
    except Exception as exc:
        logger.warning(
            "insights.source_summary.backfill_failed",
            project_id=project_id,
            error=str(exc),
        )


async def _backfill_source_doc_summaries(project_id: str, source_doc_ids: list[str]) -> None:
    """Backward-compatible wrapper for source summary backfill."""
    await ensure_source_doc_summaries(project_id, source_doc_ids)


def _to_insight_card(row: dict) -> InsightCard:
    """Map repository row to InsightCard schema."""
    return InsightCard.model_validate(
        {
            "id": str(row["id"]),
            "topic_label": str(row.get("topic_label") or ""),
            "executive_summary": str(row.get("executive_summary") or ""),
            "key_findings": [
                str(item) for item in _to_list(row.get("key_findings")) if str(item).strip()
            ],
            "trend_signal": str(row.get("trend_signal") or "unknown"),
            "cluster_size": int(row.get("cluster_size") or 0),
            "source_doc_count": int(row.get("source_doc_count") or 0),
            "full_report_status": str(row.get("full_report_status") or "pending"),
            "source_type_counts": {
                k: int(v) for k, v in _to_dict(row.get("source_type_counts")).items()
            },
        }
    )


def _to_insight_detail(row: dict) -> InsightDetail:
    """Map repository row to InsightDetail schema."""
    return InsightDetail.model_validate(
        {
            "id": str(row["id"]),
            "topic_label": str(row.get("topic_label") or ""),
            "executive_summary": str(row.get("executive_summary") or ""),
            "key_findings": [
                str(item) for item in _to_list(row.get("key_findings")) if str(item).strip()
            ],
            "trend_signal": str(row.get("trend_signal") or "unknown"),
            "contradictions": row.get("contradictions"),
            "cluster_size": int(row.get("cluster_size") or 0),
            "source_doc_count": int(row.get("source_doc_count") or 0),
            "chunk_ids": [str(item) for item in _to_list(row.get("chunk_ids")) if str(item).strip()],
            "source_doc_ids": [
                str(item) for item in _to_list(row.get("source_doc_ids")) if str(item).strip()
            ],
            "full_report": row.get("full_report"),
            "full_report_status": str(row.get("full_report_status") or "pending"),
            "source_type_counts": {
                k: int(v) for k, v in _to_dict(row.get("source_type_counts")).items()
            },
            "source_docs": _to_list(row.get("source_docs")),
        }
    )


async def list_insights(user_id: str, project_id: str) -> list[InsightCard]:
    """Return insight cards for a project, with Redis cache."""
    if not settings.INSIGHTS_ENABLED:
        return []
    await _ensure_project_owned(user_id, project_id)

    redis = await get_redis()
    cache_key = RedisKeys.INSIGHTS_CACHE.format(project_id=project_id)
    try:
        cached = await redis.get(cache_key)
        if cached:
            payload = json.loads(cached)
            if isinstance(payload, list):
                return [InsightCard.model_validate(item) for item in payload]
    except Exception:
        logger.warning("insights.list.cache_read_failed", project_id=project_id)

    rows = await insights_repo.get_insights_for_project_for_service(project_id)
    cards = [_to_insight_card(row) for row in rows]

    try:
        await redis.setex(
            cache_key,
            RedisTTL.INSIGHTS_CACHE,
            json.dumps([card.model_dump() for card in cards]),
        )
    except Exception:
        logger.warning("insights.list.cache_write_failed", project_id=project_id)
    return cards


async def get_insight_detail(user_id: str, project_id: str, insight_id: str) -> InsightDetail:
    """Return one insight detail object."""
    await _ensure_project_owned(user_id, project_id)
    row = await insights_repo.get_insight_by_id_for_service(project_id, insight_id)
    if row is None:
        raise LookupError("Insight not found")
    missing_summary_doc_ids = _missing_source_summary_doc_ids(_to_list(row.get("source_docs")))
    if settings.INSIGHTS_BACKFILL_SUMMARIES_ON_DETAIL_READ and missing_summary_doc_ids:
        await ensure_source_doc_summaries(
            project_id,
            missing_summary_doc_ids,
        )
        row = await insights_repo.get_insight_by_id_for_service(project_id, insight_id)
        if row is None:
            raise LookupError("Insight not found")
    return _to_insight_detail(row)


async def stream_full_report(
    user_id: str,
    project_id: str,
    insight_id: str,
    *,
    skip_project_ownership_check: bool = False,
    bypass_daily_cap: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream full report generation for an insight and persist on completion."""
    started_at = perf_counter()
    if not settings.INSIGHTS_ENABLED:
        insights_full_report_total.labels(status="skipped_disabled").inc()
        yield _sse_frame({"error": "Insights are temporarily disabled."})
        yield "data: [DONE]\n\n"
        return
    if not settings.INSIGHTS_FULL_REPORT_ENABLED:
        insights_full_report_total.labels(status="skipped_disabled").inc()
        yield _sse_frame({"error": "Full report generation is temporarily disabled."})
        yield "data: [DONE]\n\n"
        return

    if not skip_project_ownership_check:
        await _ensure_project_owned(user_id, project_id)
    row = await insights_repo.get_insight_by_id_for_service(project_id, insight_id)
    if row is None:
        yield _sse_frame({"error": "Insight not found."})
        yield "data: [DONE]\n\n"
        return

    detail = _to_insight_detail(row)
    if detail.full_report_status == "done" and detail.full_report:
        yield _sse_frame({"token": detail.full_report})
        yield "data: [DONE]\n\n"
        return

    redis = await get_redis()
    lock_key = RedisKeys.INSIGHT_REPORT_LOCK.format(insight_id=insight_id)
    acquired = await redis.set(lock_key, "1", ex=RedisTTL.INSIGHT_REPORT_LOCK, nx=True)
    if not acquired:
        insights_full_report_total.labels(status="lock_contention").inc()
        logger.info(
            "insights.full_report.lock_contention",
            project_id=project_id,
            insight_id=insight_id,
        )
        latest = await insights_repo.get_insight_by_id_for_service(project_id, insight_id)
        if latest:
            latest_detail = _to_insight_detail(latest)
            if latest_detail.full_report_status == "done" and latest_detail.full_report:
                yield _sse_frame({"token": latest_detail.full_report})
            else:
                yield _sse_frame({"type": "status", "status": latest_detail.full_report_status})
        else:
            yield _sse_frame({"error": "Insight not found."})
        yield "data: [DONE]\n\n"
        return

    insights_full_report_total.labels(status="started").inc()
    try:
        max_reports_per_day = (
            None
            if bypass_daily_cap
            else settings.INSIGHTS_MAX_REPORTS_PER_PROJECT_PER_DAY
        )
        claimed = await insights_repo.claim_full_report_generation_for_service(
            project_id,
            insight_id,
            max_reports_per_day,
        )
        if not claimed:
            if not bypass_daily_cap:
                generated_today = await insights_repo.count_reports_generated_today_for_service(project_id)
                if generated_today >= settings.INSIGHTS_MAX_REPORTS_PER_PROJECT_PER_DAY:
                    insights_full_report_total.labels(status="daily_cap").inc()
                    yield _sse_frame({"error": "Daily report generation limit reached for this project."})
                    yield "data: [DONE]\n\n"
                    return
            latest = await insights_repo.get_insight_by_id_for_service(project_id, insight_id)
            if latest:
                latest_detail = _to_insight_detail(latest)
                if latest_detail.full_report_status == "done" and latest_detail.full_report:
                    yield _sse_frame({"token": latest_detail.full_report})
                else:
                    yield _sse_frame({"type": "status", "status": latest_detail.full_report_status})
            yield "data: [DONE]\n\n"
            return

        all_chunks = await insights_repo.get_chunks_for_insight_for_service(
            project_id,
            detail.chunk_ids,
        )
        if detail.source_doc_ids and len(all_chunks) < max(8, settings.CLUSTER_REPORT_MAX_CHUNKS // 2):
            expanded_limit = max(80, settings.CLUSTER_REPORT_MAX_CHUNKS * 4)
            expanded_chunks = await insights_repo.get_chunks_for_documents_for_service(
                project_id,
                detail.source_doc_ids,
                limit=expanded_limit,
            )
            if all_chunks:
                all_chunks = _merge_chunk_rows(all_chunks, expanded_chunks)
                logger.info(
                    "insights.full_report.context_expand_source_docs",
                    project_id=project_id,
                    insight_id=insight_id,
                    source_doc_count=len(detail.source_doc_ids),
                    representative_chunk_count=len(detail.chunk_ids),
                    merged_chunk_count=len(all_chunks),
                    limit=expanded_limit,
                )
            else:
                all_chunks = expanded_chunks
                logger.info(
                    "insights.full_report.context_fallback_source_docs",
                    project_id=project_id,
                    insight_id=insight_id,
                    source_doc_count=len(detail.source_doc_ids),
                    chunk_count=len(all_chunks),
                    limit=expanded_limit,
                )

        if not all_chunks:
            insights_full_report_total.labels(status="failed").inc()
            duration_seconds = perf_counter() - started_at
            insights_full_report_duration_seconds.observe(duration_seconds)
            await insights_repo.set_full_report_status_for_service(project_id, insight_id, "failed")
            logger.warning(
                "insights.full_report.no_context_chunks",
                project_id=project_id,
                insight_id=insight_id,
                chunk_id_count=len(detail.chunk_ids),
                source_doc_count=len(detail.source_doc_ids),
                duration_seconds=round(duration_seconds, 3),
            )
            yield _sse_frame({"error": "No usable source chunks found for this insight."})
            yield "data: [DONE]\n\n"
            return

        assembled_parts: list[str] = []
        async for token in generate_full_report(detail.model_dump(), all_chunks):
            assembled_parts.append(token)
            yield _sse_frame({"token": token})

        full_report = "".join(assembled_parts).strip()
        if full_report:
            final_report = await refine_full_report(
                detail.model_dump(),
                full_report,
                all_chunks,
            )
            persisted_report = str(final_report or "").strip() or full_report
            persisted_report = link_evidence_references(persisted_report, all_chunks) or persisted_report
            await insights_repo.update_full_report_for_service(
                project_id,
                insight_id,
                persisted_report,
                "done",
            )
            if persisted_report != full_report:
                yield _sse_frame({"type": "replace", "report": persisted_report})
            insights_full_report_total.labels(status="success").inc()
            duration_seconds = perf_counter() - started_at
            insights_full_report_duration_seconds.observe(duration_seconds)
            logger.info(
                "insights.full_report.done",
                project_id=project_id,
                insight_id=insight_id,
                duration_seconds=round(duration_seconds, 3),
                output_chars=len(persisted_report),
                refined=bool(persisted_report != full_report),
            )
        else:
            insights_full_report_total.labels(status="failed").inc()
            duration_seconds = perf_counter() - started_at
            insights_full_report_duration_seconds.observe(duration_seconds)
            await insights_repo.set_full_report_status_for_service(project_id, insight_id, "failed")
            logger.warning(
                "insights.full_report.empty_output",
                project_id=project_id,
                insight_id=insight_id,
                duration_seconds=round(duration_seconds, 3),
            )
            yield _sse_frame({"error": "Full report generation returned empty output."})
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        insights_full_report_total.labels(status="cancelled").inc()
        await insights_repo.set_full_report_status_for_service(project_id, insight_id, "pending")
        raise
    except Exception as exc:
        insights_full_report_total.labels(status="failed").inc()
        duration_seconds = perf_counter() - started_at
        insights_full_report_duration_seconds.observe(duration_seconds)
        await insights_repo.set_full_report_status_for_service(project_id, insight_id, "failed")
        logger.exception(
            "insights.full_report.failed",
            project_id=project_id,
            insight_id=insight_id,
            error=str(exc),
            duration_seconds=round(duration_seconds, 3),
        )
        yield _sse_frame({"error": "Full report generation failed."})
        yield "data: [DONE]\n\n"
    finally:
        await redis.delete(lock_key)


async def enqueue_cluster(user_id: str, project_id: str, arq_pool: ArqRedis) -> bool:
    """Enqueue cluster_project if no active lock exists."""
    if not settings.INSIGHTS_ENABLED:
        return False
    await _ensure_project_owned(user_id, project_id)
    redis = await get_redis()
    running_lock_key = RedisKeys.CLUSTER_LOCK.format(project_id=project_id)
    enqueue_lock_key = RedisKeys.CLUSTER_ENQUEUE_LOCK.format(project_id=project_id)
    dirty_key = RedisKeys.CLUSTER_DIRTY.format(project_id=project_id)
    if await redis.exists(running_lock_key):
        await redis.set(
            dirty_key,
            "1",
            ex=_cluster_enqueue_ttl_seconds(),
        )
        logger.info(
            "insights.enqueue_cluster.marked_dirty",
            project_id=project_id,
        )
        return False
    enqueue_slot = await redis.set(
        enqueue_lock_key,
        "1",
        ex=_cluster_enqueue_ttl_seconds(),
        nx=True,
    )
    if not enqueue_slot:
        return False
    try:
        await arq_pool.enqueue_job("cluster_project_task", project_id)
        return True
    except Exception:
        await redis.delete(enqueue_lock_key)
        raise
