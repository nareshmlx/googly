"""Cluster follow-up service with strict source-scoped context only."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import datetime
from time import perf_counter
from typing import Any

import numpy as np
import structlog
from agno.run.agent import RunContentEvent

from app.agents.cluster_followup import build_cluster_followup_agent
from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.metrics import insights_followup_requests_total
from app.core.redis import get_redis
from app.core.serialization import cosine_similarity as _cosine_similarity
from app.core.serialization import parse_embedding as _parse_embedding
from app.core.serialization import sse_frame as _sse_frame
from app.core.serialization import to_dict as _to_dict
from app.core.serialization import to_list as _to_list
from app.kb.embedder import embed_texts
from app.models.schemas import FollowupMessage
from app.repositories import cluster_followup_history as followup_repo
from app.repositories import insights as insights_repo
from app.services import project as project_service

logger = structlog.get_logger(__name__)


def _build_context(chunks: list[dict]) -> str:
    """Build model context text from chunk rows."""
    parts: list[str] = []
    for idx, chunk in enumerate(chunks[:40], start=1):
        metadata = _to_dict(chunk.get("metadata"))
        title = str(chunk.get("title") or "Untitled")
        url = str(metadata.get("url") or "").strip()
        header = f"{idx}. {title}"
        if url:
            header = f"{header} ({url})"
        content = str(chunk.get("content") or "").strip()
        if content:
            parts.append(f"{header}\n{content[:1400]}")
    return "\n\n".join(parts)


def _history_cache_key(insight_id: str, user_id: str) -> str:
    """Build Redis key for per-insight per-user follow-up history."""
    return RedisKeys.FOLLOWUP_HISTORY.format(insight_id=insight_id, user_id=user_id)


def _row_to_followup_message(row: dict[str, Any]) -> FollowupMessage:
    """Normalize repository/cache row shape into FollowupMessage."""
    created_at_raw = row.get("created_at")
    if isinstance(created_at_raw, datetime):
        created_at = created_at_raw.isoformat()
    else:
        created_at = str(created_at_raw or "")
    return FollowupMessage.model_validate(
        {
            "id": str(row["id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "context_source": str(row.get("context_source") or "cluster"),
            "created_at": created_at,
        }
    )


async def _user_project_ids(user_id: str) -> list[str]:
    """Return all project IDs owned by a user."""
    projects = await project_service.list_projects(user_id)
    ids: list[str] = []
    for project in projects:
        project_id = str(project.get("id") or "").strip()
        if project_id:
            ids.append(project_id)
    return ids


async def stream_cluster_followup(
    user_id: str,
    insight_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    """Stream a follow-up answer for one insight using strict source-scoped context only."""
    started_at = perf_counter()
    if not settings.INSIGHTS_ENABLED or not settings.INSIGHTS_FOLLOWUP_ENABLED:
        insights_followup_requests_total.labels(context_source="disabled").inc()
        yield _sse_frame({"error": "Insight follow-up is temporarily disabled."})
        yield "data: [DONE]\n\n"
        return

    project_ids = await _user_project_ids(user_id)
    insight = await insights_repo.get_insight_by_id_in_projects_for_service(project_ids, insight_id)
    if insight is None:
        yield _sse_frame({"error": "Insight not found."})
        yield "data: [DONE]\n\n"
        return

    project_id = str(insight["project_id"])

    chunk_ids = [str(item) for item in _to_list(insight.get("chunk_ids")) if str(item).strip()]
    source_doc_ids = [
        str(item) for item in _to_list(insight.get("source_doc_ids")) if str(item).strip()
    ]

    cluster_chunks = await insights_repo.get_chunks_for_insight_for_service(project_id, chunk_ids)

    embedded = await embed_texts([message])
    if not embedded:
        yield _sse_frame({"error": "Unable to embed follow-up query."})
        yield "data: [DONE]\n\n"
        return
    query_vector = np.asarray(embedded[0], dtype=np.float32)
    expected_dim = int(query_vector.size)
    best_score = 0.0
    for chunk in cluster_chunks:
        emb = _parse_embedding(chunk.get("embedding"), expected_dim=expected_dim)
        if emb is None:
            continue
        best_score = max(best_score, _cosine_similarity(query_vector, emb))

    context_source = "cluster"
    context_chunks = cluster_chunks
    if best_score < settings.KB_SCORE_THRESHOLD:
        context_chunks = await insights_repo.get_chunks_for_documents_for_service(
            project_id,
            source_doc_ids,
            limit=max(80, settings.CLUSTER_REPORT_MAX_CHUNKS * 4),
        )
        context_source = "cluster_docs_expanded"

    if not context_chunks:
        insights_followup_requests_total.labels(context_source="no_context").inc()
        yield _sse_frame({"type": "no_context"})
        yield "data: [DONE]\n\n"
        return

    insights_followup_requests_total.labels(context_source=context_source).inc()
    yield _sse_frame({"type": "context_source", "source": context_source})
    context = _build_context(context_chunks)

    agent = build_cluster_followup_agent()
    prompt = (
        f"Question: {message.strip()}\n\n"
        "Use only this context to answer:\n\n"
        f"{context}\n\n"
        "If context is insufficient, say so briefly."
    )

    assistant_parts: list[str] = []
    try:
        async for event in agent.arun(prompt, stream=True, stream_events=True):
            if isinstance(event, RunContentEvent):
                token = event.content
                if token and isinstance(token, str):
                    assistant_parts.append(token)
                    yield _sse_frame({"token": token})
    except Exception:
        insights_followup_requests_total.labels(context_source="error").inc()
        logger.exception(
            "cluster_followup.stream_failed",
            project_id=project_id,
            insight_id=insight_id,
        )
        yield _sse_frame({"error": "Follow-up generation failed."})
        yield "data: [DONE]\n\n"
        return

    assistant_text = "".join(assistant_parts).strip()
    if assistant_text:
        try:
            await followup_repo.append_messages_for_service(
                insight_id=insight_id,
                project_id=project_id,
                user_id=user_id,
                user_msg=message,
                assistant_msg=assistant_text,
                context_source=context_source,
            )
            rows = await followup_repo.get_history_for_service(
                insight_id=insight_id,
                project_id=project_id,
                user_id=user_id,
            )
            messages = [_row_to_followup_message(row) for row in rows]
            redis = await get_redis()
            await redis.setex(
                _history_cache_key(insight_id, user_id),
                RedisTTL.FOLLOWUP_HISTORY,
                json.dumps([message.model_dump() for message in messages]),
            )
        except Exception:
            logger.warning(
                "cluster_followup.persist_or_cache_failed",
                project_id=project_id,
                insight_id=insight_id,
            )

    logger.info(
        "cluster_followup.done",
        project_id=project_id,
        insight_id=insight_id,
        context_source=context_source,
        duration_seconds=round(perf_counter() - started_at, 3),
    )

    yield "data: [DONE]\n\n"


async def get_followup_history(
    user_id: str,
    insight_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[FollowupMessage]:
    """Return follow-up history for one insight scoped to the requesting user."""
    project_ids = await _user_project_ids(user_id)
    insight = await insights_repo.get_insight_by_id_in_projects_for_service(project_ids, insight_id)
    if insight is None:
        raise LookupError("Insight not found")

    project_id = str(insight["project_id"])

    # Skip cache if we are using an offset, to ensure freshness and correct paging.
    # Alternatively, cache the whole history and slice it here, but current repo does
    # the slicing in SQL which is better for deep history.
    if offset == 0:
        cache_key = _history_cache_key(insight_id, user_id)
        try:
            redis = await get_redis()
            cached = await redis.get(cache_key)
            if cached:
                payload = json.loads(cached)
                if isinstance(payload, list):
                    return [FollowupMessage.model_validate(item) for item in payload[:limit]]
        except Exception:
            logger.warning(
                "cluster_followup.history_cache_read_failed",
                project_id=project_id,
                insight_id=insight_id,
            )

    rows = await followup_repo.get_history_for_service(
        insight_id=insight_id,
        project_id=project_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    messages = [_row_to_followup_message(row) for row in rows]

    # Only cache the first page
    if offset == 0 and messages:
        try:
            redis = await get_redis()
            cache_key = _history_cache_key(insight_id, user_id)
            await redis.setex(
                cache_key,
                RedisTTL.FOLLOWUP_HISTORY,
                json.dumps([message.model_dump() for message in messages]),
            )
        except Exception:
            logger.warning(
                "cluster_followup.history_cache_write_failed",
                project_id=project_id,
                insight_id=insight_id,
            )
    return messages

