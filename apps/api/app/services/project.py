"""ProjectService — orchestrates project CRUD, intent extraction, and ARQ enqueueing.

Rule: This file contains business logic only. No direct SQL. No HTTP.
DB access goes through app.repositories.project. ARQ enqueuing goes through
the ARQ pool. Intent extraction goes through app.kb.intent_extractor.
"""

import asyncio
import json
from datetime import UTC, datetime

import structlog
from arq import ArqRedis
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.db import get_db_pool
from app.core.redis import get_redis
from app.kb.embedder import embed_texts
from app.kb.intent_extractor import extract_intent
from app.repositories import project as project_repo

logger = structlog.get_logger(__name__)


async def _invalidate_projects_summary_cache(user_id: str) -> None:
    """Invalidate cached project summaries for one user."""
    try:
        redis = await get_redis()
        await redis.delete(RedisKeys.PROJECTS_SUMMARY.format(user_id=user_id))
    except Exception:
        logger.warning("project_service.projects_summary_cache_invalidate_failed", user_id=user_id)


async def set_project_ingest_status(
    project_id: str,
    *,
    status: str,
    message: str | None = None,
    queued_at: str | None = None,
    started_at: str | None = None,
    updated_at: str | None = None,
    finished_at: str | None = None,
    job_id: str | None = None,
    total_chunks: int | None = None,
    source_counts: dict | None = None,
) -> None:
    """Persist project ingest status in Redis for API/UI visibility."""
    payload = {
        "project_id": project_id,
        "status": status,
        "message": message,
        "queued_at": queued_at,
        "started_at": started_at,
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
        "finished_at": finished_at,
        "job_id": job_id,
        "total_chunks": total_chunks,
        "source_counts": source_counts or {},
    }
    try:
        redis = await get_redis()
        key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)
        await redis.setex(key, RedisTTL.PROJECT_INGEST_STATUS.value, json.dumps(payload))
    except Exception:
        logger.warning("project_service.ingest_status_set_failed", project_id=project_id)


async def get_project_ingest_status(project_id: str) -> dict:
    """Read project ingest status from Redis with a DB-backed fallback."""
    redis_payload: dict | None = None
    try:
        redis = await get_redis()
        key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)
        raw = await redis.get(key)
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                redis_payload = parsed
    except Exception:
        logger.warning("project_service.ingest_status_get_failed", project_id=project_id)

    pool = await get_db_pool()
    row = await project_repo.fetch_project_by_id(pool, project_id)
    if not row:
        return {
            "project_id": project_id,
            "status": "not_found",
            "message": "Project not found",
            "source_counts": {},
        }

    kb_chunk_count = int(row.get("kb_chunk_count") or 0)
    refreshed = row.get("last_refreshed_at")
    refreshed_at = refreshed.isoformat() if refreshed else None

    if redis_payload:
        if kb_chunk_count > 0 and redis_payload.get("status") in {"queued", "running", "empty"}:
            redis_payload["status"] = "ready"
            redis_payload["total_chunks"] = kb_chunk_count
            redis_payload["finished_at"] = refreshed_at or redis_payload.get("finished_at")
            redis_payload["message"] = "Ingestion complete."
        redis_payload.setdefault("project_id", project_id)
        redis_payload.setdefault("source_counts", {})
        return redis_payload

    return {
        "project_id": project_id,
        "status": "ready" if kb_chunk_count > 0 else "unknown",
        "message": "Ingestion complete." if kb_chunk_count > 0 else "No recent ingest status found.",
        "queued_at": None,
        "started_at": None,
        "updated_at": refreshed_at,
        "finished_at": refreshed_at,
        "job_id": None,
        "total_chunks": kb_chunk_count if kb_chunk_count > 0 else None,
        "source_counts": {},
    }


async def _expand_description(description: str) -> str:
    """
    Expand a short project description to improve intent extraction quality.

    Returns the original description when expansion fails, so project creation
    remains resilient.
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Expand short project descriptions into 2-3 rich sentences "
                        "for research intent extraction. Include domain context, "
                        "core topics, and likely information needs. Output only the "
                        "expanded description."
                    ),
                },
                {"role": "user", "content": description},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        expanded = (response.choices[0].message.content or "").strip()

        if not expanded:
            logger.warning(
                "project_service.expand_description.empty_response",
                description_len=len(description),
            )
            return description

        logger.info(
            "project_service.expand_description.success",
            description_len=len(description),
            expanded_len=len(expanded),
        )
        return expanded
    except Exception as exc:
        logger.warning(
            "project_service.expand_description.failed",
            description_preview=description[:80],
            error=str(exc),
        )
        return description


async def create_project(
    user_id: str,
    title: str,
    description: str,
    refresh_strategy: str,
    arq_pool: ArqRedis,
    tiktok_enabled: bool = True,
    instagram_enabled: bool = True,
    papers_enabled: bool = True,
    patents_enabled: bool = True,
    perigon_enabled: bool = True,
    tavily_enabled: bool = True,
    exa_enabled: bool = True,
    openalex_enabled: bool | None = None,
) -> dict:
    """
    Create a project, extract structured intent, and enqueue initial KB ingestion.

    Intent extraction runs synchronously here (not in the worker) so the response
    to the client already contains meaningful structured_intent — the UI can show
    the extracted keywords immediately without polling.

    Enqueueing ingest_project is fire-and-forget: the worker runs after this
    returns. Project creation does not wait for KB population.
    """
    pool = await get_db_pool()

    logger.info("project_service.create.start", user_id=user_id, title=title)

    if openalex_enabled is not None:
        papers_enabled = openalex_enabled

    # 1. Insert with empty intent first — get a real project_id
    project = await project_repo.insert_project(
        pool,
        user_id,
        title,
        description,
        refresh_strategy,
        tiktok_enabled=tiktok_enabled,
        instagram_enabled=instagram_enabled,
        papers_enabled=papers_enabled,
        patents_enabled=patents_enabled,
        perigon_enabled=perigon_enabled,
        tavily_enabled=tavily_enabled,
        exa_enabled=exa_enabled,
    )
    project_id = project["id"]
    logger.info("project_service.create.inserted", project_id=project_id)

    # 2. Expand and extract intent synchronously — safe to fail
    try:
        expanded = await asyncio.wait_for(
            _expand_description(description),
            timeout=settings.PROJECT_CREATE_INTENT_TIMEOUT,
        )
    except TimeoutError:
        logger.warning(
            "project_service.create.expand_timeout",
            project_id=project_id,
            timeout_seconds=settings.PROJECT_CREATE_INTENT_TIMEOUT,
        )
        expanded = description

    try:
        intent = await asyncio.wait_for(
            extract_intent(expanded),
            timeout=settings.PROJECT_CREATE_INTENT_TIMEOUT,
        )
    except TimeoutError:
        logger.warning(
            "project_service.create.intent_timeout",
            project_id=project_id,
            timeout_seconds=settings.PROJECT_CREATE_INTENT_TIMEOUT,
        )
        fallback = description[:200]
        intent = {
            "domain": "beauty_market_intelligence",
            "keywords": [token for token in fallback.split()[:8] if token.strip()],
            "search_filters": {
                "news": fallback,
                "papers": fallback,
                "patents": fallback,
                "tiktok": "",
                "social": "",
                "instagram": "",
            },
            "confidence": 0.0,
        }
    logger.info(
        "project_service.create.intent_extracted",
        project_id=project_id,
        domain=intent.get("domain"),
    )

    # 3. Store intent
    await project_repo.update_project_intent(pool, project_id, intent)
    project["structured_intent"] = intent

    embedding_input = f"{expanded}\n{json.dumps(intent, sort_keys=True)}"
    try:
        vectors = await embed_texts([embedding_input])
        if vectors:
            await project_repo.update_project_intent_embedding(pool, project_id, vectors[0])
            logger.info(
                "project_service.create.intent_embedding_stored",
                project_id=project_id,
                embedding_dim=len(vectors[0]),
            )
    except Exception as exc:
        logger.warning(
            "project_service.create.intent_embedding_failed",
            project_id=project_id,
            error=str(exc),
        )

    await _invalidate_projects_summary_cache(user_id)

    # 4. Enqueue initial KB population (non-blocking)
    try:
        job = await arq_pool.enqueue_job("ingest_project", project_id)
        now = datetime.now(UTC).isoformat()
        await set_project_ingest_status(
            project_id,
            status="queued",
            message="Ingestion queued.",
            queued_at=now,
            updated_at=now,
            job_id=str(getattr(job, "job_id", "") or ""),
        )
        logger.info("project_service.create.ingestion_enqueued", project_id=project_id)
    except Exception:
        # ARQ failure must not fail project creation — KB will be empty until user retries
        logger.warning("project_service.create.enqueue_failed", project_id=project_id)

    return project


async def list_projects(user_id: str) -> list[dict]:
    """Return all projects for a user, newest first."""
    pool = await get_db_pool()
    return await project_repo.list_projects(pool, user_id)


async def get_project(project_id: str, user_id: str) -> dict | None:
    """
    Fetch a single project scoped to its owner.

    Returns None if the project does not exist or the user does not own it.
    Caller is responsible for returning 404.
    """
    pool = await get_db_pool()
    return await project_repo.fetch_project(pool, project_id, user_id)


async def get_discover_feed(project_id: str) -> list[dict]:
    """Fetch discover rows for all enabled source types.

    Returns an empty list when no content has been ingested yet.
    """
    pool = await get_db_pool()
    rows = await project_repo.fetch_discover_feed(pool, project_id)
    logger.info(
        "project_service.discover_feed.fetched",
        project_id=project_id,
        total=len(rows),
        instagram=sum(1 for r in rows if r["source"] == "social_instagram"),
        tiktok=sum(1 for r in rows if r["source"] == "social_tiktok"),
    )
    return rows


async def delete_project(project_id: str, user_id: str) -> bool:
    """
    Delete a project and all its knowledge chunks (cascade).

    Returns True if deleted, False if not found / not owned.
    """
    pool = await get_db_pool()
    deleted = await project_repo.delete_project(pool, project_id, user_id)
    if deleted:
        logger.info("project_service.deleted", project_id=project_id, user_id=user_id)
        await _invalidate_projects_summary_cache(user_id)
    return deleted
