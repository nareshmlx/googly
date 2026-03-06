"""ProjectService — orchestrates project CRUD, intent extraction, and ARQ enqueueing.

Rule: This file contains business logic only. No direct SQL. No HTTP.
DB access goes through app.repositories.project. ARQ enqueuing goes through
the ARQ pool. Intent extraction goes through app.kb.intent_extractor.
"""

import asyncio
import json
import re
from datetime import UTC, datetime

import structlog
from arq import ArqRedis
from arq.jobs import deserialize_result

from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.redis import get_redis
from app.core.utils import build_stable_signature, parse_metadata
from app.kb.embedder import embed_texts
from app.kb.intent_extractor import extract_intent
from app.repositories import project as project_repo
from app.repositories import source_asset as source_asset_repo
from app.services import project_wizard as project_wizard_service
from app.tools.llm import expand_project_description

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
    source_diagnostics: dict | None = None,
    fulltext_enqueued: int = 0,
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
        "source_diagnostics": source_diagnostics or {},
        "fulltext_enqueued": int(fulltext_enqueued),
    }
    try:
        redis = await get_redis()
        key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)
        await redis.setex(key, RedisTTL.PROJECT_INGEST_STATUS, json.dumps(payload))
    except Exception:
        logger.warning("project_service.ingest_status_set_failed", project_id=project_id)


async def get_project_ingest_status(project_id: str) -> dict:
    """Read project ingest status from Redis with a DB-backed fallback."""
    redis_payload: dict | None = None
    redis_client = None
    status_key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)
    try:
        redis_client = await get_redis()
        raw = await redis_client.get(status_key)
        if raw:
            redis_payload = parse_metadata(raw)
    except Exception:
        logger.warning("project_service.ingest_status_get_failed", project_id=project_id)

    row = await project_repo.fetch_project_by_id_for_service(project_id)
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
    enrichment = await source_asset_repo.fetch_project_enrichment_counts_for_service(project_id)

    if redis_payload:
        if (
            redis_payload.get("status") in {"queued", "running"}
            and redis_client is not None
            and str(redis_payload.get("job_id") or "").strip()
        ):
            job_id = str(redis_payload.get("job_id") or "").strip()
            try:
                result_key = f"arq:result:{job_id}"
                result_raw = await redis_client.get(result_key)
                if result_raw:
                    now = datetime.now(UTC).isoformat()
                    result = deserialize_result(result_raw)
                    success = bool(getattr(result, "success", False))
                    redis_payload["status"] = (
                        "ready"
                        if success and kb_chunk_count > 0
                        else "empty"
                        if success
                        else "failed"
                    )
                    redis_payload["total_chunks"] = kb_chunk_count if kb_chunk_count > 0 else 0
                    redis_payload["finished_at"] = (
                        redis_payload.get("finished_at") or refreshed_at or now
                    )
                    redis_payload["updated_at"] = now
                    if success:
                        redis_payload["message"] = (
                            "Ingestion complete."
                            if kb_chunk_count > 0
                            else "Ingestion finished with no documents. Check source API connectivity/keys."
                        )
                    else:
                        redis_payload["message"] = (
                            "Ingestion failed in background worker. Check worker logs."
                        )
                    await redis_client.setex(
                        status_key,
                        RedisTTL.PROJECT_INGEST_STATUS,
                        json.dumps(redis_payload),
                    )
            except Exception:
                logger.warning(
                    "project_service.ingest_status_arq_result_check_failed",
                    project_id=project_id,
                )

        pending_enrichment = int(enrichment.get("pending") or 0)
        if pending_enrichment > 0 and redis_payload.get("status") in {"ready", "empty"}:
            redis_payload["status"] = "running"
            redis_payload["message"] = (
                f"Main ingestion complete. Fulltext enrichment in progress ({pending_enrichment} pending)."
            )

        if (
            redis_payload.get("status") in {"ready", "empty"}
            and not redis_payload.get("total_chunks")
            and kb_chunk_count > 0
        ):
            redis_payload["total_chunks"] = kb_chunk_count
            redis_payload["finished_at"] = refreshed_at or redis_payload.get("finished_at")

        redis_payload.setdefault("project_id", project_id)
        redis_payload.setdefault("source_counts", {})
        redis_payload.setdefault("source_diagnostics", {})
        redis_payload["enrichment"] = enrichment
        return redis_payload

    return {
        "project_id": project_id,
        "status": "ready" if kb_chunk_count > 0 else "unknown",
        "message": "Ingestion complete."
        if kb_chunk_count > 0
        else "No recent ingest status found.",
        "queued_at": None,
        "started_at": None,
        "updated_at": refreshed_at,
        "finished_at": refreshed_at,
        "job_id": None,
        "total_chunks": kb_chunk_count if kb_chunk_count > 0 else None,
        "source_counts": {},
        "source_diagnostics": {},
        "enrichment": enrichment,
    }


def _canonicalize_upload_ids(upload_ids: list[str] | None) -> list[str]:
    """Normalize upload IDs to deterministic sorted unique strings."""
    values = [str(value).strip() for value in (upload_ids or []) if str(value).strip()]
    return sorted(set(values))


def _upload_signature(upload_ids: list[str]) -> str:
    """Compute stable signature for idempotent bootstrap runs."""
    return build_stable_signature(upload_ids)


async def set_project_setup_status(
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
    """Persist project bootstrap/setup status to Redis and project metadata."""
    canonical_upload_ids = _canonicalize_upload_ids(upload_ids)
    payload = {
        "project_id": project_id,
        "status": status,
        "phase": phase,
        "progress_percent": max(0, min(100, int(progress_percent))),
        "message": message,
        "updated_at": datetime.now(UTC).isoformat(),
        "error": error,
        "upload_ids": canonical_upload_ids,
        "upload_signature": upload_signature,
        "job_id": job_id,
    }
    try:
        redis = await get_redis()
        key = RedisKeys.PROJECT_SETUP_STATUS.format(project_id=project_id)
        await redis.setex(key, RedisTTL.PROJECT_SETUP_STATUS, json.dumps(payload))
    except Exception:
        logger.warning("project_service.setup_status_set_redis_failed", project_id=project_id)

    try:
        await project_repo.update_project_setup_status_metadata_for_service(project_id, payload)
    except Exception:
        logger.warning("project_service.setup_status_set_metadata_failed", project_id=project_id)
    return payload


async def get_project_setup_status(project_id: str) -> dict:
    """Read project setup status from Redis with metadata fallback."""
    try:
        redis = await get_redis()
        raw = await redis.get(RedisKeys.PROJECT_SETUP_STATUS.format(project_id=project_id))
        if raw:
            return parse_metadata(raw)
    except Exception:
        logger.warning("project_service.setup_status_get_redis_failed", project_id=project_id)

    try:
        row = await project_repo.fetch_project_by_id_for_service(project_id)
        if not row:
            return {
                "project_id": project_id,
                "status": "not_found",
                "phase": "unknown",
                "progress_percent": 0,
                "message": "Project not found",
                "upload_ids": [],
                "upload_signature": "",
                "job_id": None,
            }
        metadata = parse_metadata(row.get("metadata"))
        setup = metadata.get("setup_status")
        if isinstance(setup, dict):
            return setup
    except Exception:
        logger.warning("project_service.setup_status_get_metadata_failed", project_id=project_id)

    return {
        "project_id": project_id,
        "status": "unknown",
        "phase": "pending",
        "progress_percent": 0,
        "message": "No setup status found.",
        "upload_ids": [],
        "upload_signature": "",
        "job_id": None,
    }


async def enqueue_project_bootstrap(
    *,
    project_id: str,
    user_id: str,
    upload_ids: list[str] | None,
    arq_pool: ArqRedis,
) -> dict:
    """Enqueue bootstrap setup task with idempotency for active runs."""
    canonical_upload_ids = _canonicalize_upload_ids(upload_ids)
    signature = _upload_signature(canonical_upload_ids)
    current = await get_project_setup_status(project_id)
    current_status = str(current.get("status") or "").lower()
    current_signature = str(current.get("upload_signature") or "")

    if current_status in {"queued", "running"} and current_signature == signature:
        return current

    job = await arq_pool.enqueue_job(
        "bootstrap_project_setup",
        project_id,
        user_id,
        canonical_upload_ids,
    )
    payload = await set_project_setup_status(
        project_id,
        status="queued",
        phase="queued",
        progress_percent=0,
        message="Project setup queued.",
        upload_ids=canonical_upload_ids,
        upload_signature=signature,
        job_id=str(getattr(job, "job_id", "") or ""),
    )
    return payload


async def _expand_description(description: str) -> str:
    """
    Expand a short project description to improve intent extraction quality.

    Returns the original description when expansion fails, so project creation
    remains resilient.
    """
    expanded = await expand_project_description(description)
    if not expanded:
        return description
    logger.info(
        "project_service.expand_description.success",
        description_len=len(description),
        expanded_len=len(expanded),
    )
    return expanded


def _intent_keywords(structured_intent: dict | None, *, max_items: int = 40) -> list[str]:
    if not isinstance(structured_intent, dict):
        return []

    out: list[str] = []
    seen: set[str] = set()

    def _push(value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(text)

    for field in ("keywords", "must_match_terms", "entities", "domain_terms"):
        values = structured_intent.get(field)
        if not isinstance(values, list | tuple | set):
            continue
        for item in values:
            _push(item)
            if len(out) >= max_items:
                return out

    domain = str(structured_intent.get("domain") or "").replace("_", " ").strip()
    if domain:
        _push(domain)

    return out[:max_items]


def _build_hybrid_embedding_text(base_text: str, structured_intent: dict | None) -> str:
    primary = str(base_text or "").strip()
    keywords = _intent_keywords(structured_intent)
    if keywords:
        keywords_block = ", ".join(keywords)
        if primary:
            return f"{primary}\n\nIntent Keywords: {keywords_block}"
        return f"Intent Keywords: {keywords_block}"

    if primary:
        return primary
    if isinstance(structured_intent, dict):
        return json.dumps(structured_intent, sort_keys=True)
    return ""


async def create_project(
    user_id: str,
    title: str,
    description: str,
    refresh_strategy: str,
    arq_pool: ArqRedis,
    tiktok_enabled: bool = True,
    instagram_enabled: bool = True,
    youtube_enabled: bool = True,
    reddit_enabled: bool = True,
    x_enabled: bool = True,
    papers_enabled: bool = True,
    patents_enabled: bool = True,
    perigon_enabled: bool = True,
    tavily_enabled: bool = True,
    exa_enabled: bool = True,
    openalex_enabled: bool | None = None,
    enqueue_ingestion: bool = True,
    skip_description_expansion: bool = False,
    intent_seed_text: str | None = None,
) -> dict:
    """
    Create a project, extract structured intent, and enqueue initial KB ingestion.

    Intent extraction runs synchronously here (not in the worker) so the response
    to the client already contains meaningful structured_intent — the UI can show
    the extracted keywords immediately without polling.

    Enqueueing ingest_project is fire-and-forget: the worker runs after this
    returns. Project creation does not wait for KB population.
    """
    logger.info("project_service.create.start", user_id=user_id, title=title)

    if openalex_enabled is not None:
        papers_enabled = openalex_enabled

    # 1. Insert with empty intent first — get a real project_id
    project = await project_repo.insert_project_for_service(
        user_id,
        title,
        description,
        refresh_strategy,
        tiktok_enabled=tiktok_enabled,
        instagram_enabled=instagram_enabled,
        youtube_enabled=youtube_enabled,
        reddit_enabled=reddit_enabled,
        x_enabled=x_enabled,
        papers_enabled=papers_enabled,
        patents_enabled=patents_enabled,
        perigon_enabled=perigon_enabled,
        tavily_enabled=tavily_enabled,
        exa_enabled=exa_enabled,
    )
    project_id = project["id"]
    logger.info("project_service.create.inserted", project_id=project_id)

    # 2. Expand and extract intent synchronously — safe to fail
    seed_text = str(intent_seed_text or "").strip() or description
    if skip_description_expansion:
        expanded = seed_text
    else:
        try:
            expanded = await asyncio.wait_for(
                _expand_description(seed_text),
                timeout=settings.PROJECT_CREATE_INTENT_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "project_service.create.expand_timeout",
                project_id=project_id,
                timeout_seconds=settings.PROJECT_CREATE_INTENT_TIMEOUT,
            )
            expanded = seed_text

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
        fallback_terms = [
            token.lower() for token in re.findall(r"[a-zA-Z0-9]+", fallback) if len(token) >= 4
        ]
        fallback_terms = list(dict.fromkeys(fallback_terms))[:6]
        fallback_tiktok = " ".join(f"#{token}" for token in fallback_terms[:4]).strip()
        fallback_instagram = " ".join(fallback_terms[:3]).strip()
        fallback_domain = "_".join(fallback_terms[:3]) or "general"
        intent = {
            "domain": fallback_domain,
            "keywords": [token for token in fallback.split()[:8] if token.strip()],
            "search_filters": {
                "news": fallback,
                "papers": fallback,
                "patents": fallback,
                "tiktok": fallback_tiktok,
                "social": fallback_tiktok,
                "instagram": fallback_instagram,
            },
            "confidence": 0.0,
        }
    logger.info(
        "project_service.create.intent_extracted",
        project_id=project_id,
        domain=intent.get("domain"),
    )

    # 3. Store intent
    await project_repo.update_project_intent_for_service(project_id, intent)
    project["structured_intent"] = intent

    embedding_input = _build_hybrid_embedding_text(expanded, intent)
    try:
        vectors = await embed_texts([embedding_input])
        if vectors:
            await project_repo.update_project_intent_embedding_for_service(project_id, vectors[0])
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
    if enqueue_ingestion:
        await enqueue_project_ingestion(project_id, arq_pool)

    return project


async def enqueue_project_ingestion(project_id: str, arq_pool: ArqRedis) -> bool:
    """Enqueue background ingest job and persist queued status."""
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
        return True
    except Exception:
        # ARQ failure must not fail project creation — KB will be empty until user retries
        logger.warning("project_service.create.enqueue_failed", project_id=project_id)
        return False


async def list_projects(user_id: str) -> list[dict]:
    """Return all projects for a user, newest first."""
    return await project_repo.list_projects_for_service(user_id)


async def get_project(project_id: str, user_id: str) -> dict | None:
    """
    Fetch a single project scoped to its owner.

    Returns None if the project does not exist or the user does not own it.
    Caller is responsible for returning 404.
    """
    return await project_repo.fetch_project_for_service(project_id, user_id)


async def get_discover_feed(project_id: str) -> list[dict]:
    """Fetch discover rows for all enabled source types.

    Returns an empty list when no content has been ingested yet.
    """
    rows = await project_repo.fetch_discover_feed_for_service(project_id)
    logger.info(
        "project_service.discover_feed.fetched",
        project_id=project_id,
        total=len(rows),
        instagram=sum(1 for r in rows if r["source"] == "social_instagram"),
        tiktok=sum(1 for r in rows if r["source"] == "social_tiktok"),
        youtube=sum(1 for r in rows if r["source"] == "social_youtube"),
        reddit=sum(1 for r in rows if r["source"] == "social_reddit"),
        x=sum(1 for r in rows if r["source"] == "social_x"),
        paper=sum(1 for r in rows if r["source"] == "paper"),
        patent=sum(1 for r in rows if r["source"] == "patent"),
        news=sum(1 for r in rows if r["source"] == "news"),
        search=sum(1 for r in rows if r["source"] == "search"),
    )
    return rows


async def apply_wizard_overrides(
    *,
    project_id: str,
    enriched_description: str,
    overrides: dict,
) -> dict | None:
    """
    Apply wizard Phase-2 overrides with schema-preserving value enhancement.

    Returns the updated project row or None when project does not exist.
    """
    project = await project_repo.fetch_project_by_id_for_service(project_id)
    if not project:
        return None

    base_intent = parse_metadata(project.get("structured_intent"))
    merged_intent = await project_wizard_service.merge_intent_with_overrides(
        structured_intent=base_intent,
        enriched_description=enriched_description,
        overrides=overrides,
    )

    await project_repo.update_project_intent_for_service(project_id, merged_intent)
    await project_repo.update_project_enriched_description_for_service(
        project_id,
        enriched_description or None,
    )

    embedding_text = (
        enriched_description
        or project.get("enriched_description")
        or project.get("description")
        or ""
    )
    embedding_input = _build_hybrid_embedding_text(embedding_text, merged_intent)
    try:
        vectors = await embed_texts([embedding_input])
        if vectors:
            await project_repo.update_project_intent_embedding_for_service(project_id, vectors[0])
    except Exception as exc:
        logger.warning(
            "project_service.apply_wizard_overrides.intent_embedding_failed",
            project_id=project_id,
            error=str(exc),
        )

    await _invalidate_projects_summary_cache(str(project.get("user_id") or ""))
    updated = await project_repo.fetch_project_by_id_for_service(project_id)
    return updated


async def delete_project(project_id: str, user_id: str) -> bool:
    """
    Delete a project and all its knowledge chunks (cascade).

    Returns True if deleted, False if not found / not owned.
    """
    deleted = await project_repo.delete_project_for_service(project_id, user_id)
    if deleted:
        logger.info("project_service.deleted", project_id=project_id, user_id=user_id)
        await _invalidate_projects_summary_cache(user_id)
    return deleted
