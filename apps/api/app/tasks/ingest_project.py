"""ingest_project ARQ task — initial KB population from enabled sources.

Supports independently toggleable ingestion sources:
  - Social: Instagram, TikTok, YouTube, Reddit, X
  - Papers: OpenAlex, Semantic Scholar, PubMed, arXiv
  - Patents: PatentsView, Lens
  - Discovery: Perigon news, Tavily web, Exa web
"""

import asyncio
import json
from datetime import UTC, datetime

import structlog

from app.core.cache_invalidation import invalidate_project_caches
from app.core.config import settings
from app.core.constants import RedisKeys
from app.core.db import get_db_pool
from app.core.redis import get_redis
from app.kb.embedder import embed_texts
from app.kb.ingester import ingest_documents
from app.repositories import project as project_repo
from app.tasks.ingest_discovery import (
    _ingest_news,
    _ingest_web_exa,
    _ingest_web_tavily,
)
from app.tasks.ingest_documents_handler import (
    _prepare_kept_documents_for_source,
    _schedule_fulltext_enrichment,
    _set_ingest_status,
)
from app.tasks.ingest_papers import (
    _ingest_papers_arxiv,
    _ingest_papers_openalex,
    _ingest_papers_pubmed,
    _ingest_papers_semantic_scholar,
)
from app.tasks.ingest_patents import (
    _ingest_patents_lens,
    _ingest_patents_patentsview,
    _ingest_patents_web_fallback,
)
from app.tasks.ingest_social_instagram import _ingest_instagram
from app.tasks.ingest_social_reddit import _ingest_reddit
from app.tasks.ingest_social_tiktok import _ingest_tiktok
from app.tasks.ingest_social_x import _ingest_x
from app.tasks.ingest_social_youtube import _ingest_youtube
from app.tasks.ingest_utils import (
    _extract_expansion_links,
    _noop,
    _project_anchor_terms,
    _query_for_social,
    _run_source_with_timeout,
)

logger = structlog.get_logger(__name__)


async def ingest_project(ctx: dict, project_id: str) -> None:
    """Populate a project's KB from all enabled sources."""
    redis = await get_redis()
    started_at = datetime.now(UTC).isoformat()
    queued_at = None
    job_id: str | None = None
    try:
        key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)
        raw = await redis.get(key)
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                queued_at = parsed.get("queued_at")
                job_id = str(parsed.get("job_id") or "").strip() or None
    except Exception:
        logger.warning("ingest_project.status_read_failed", project_id=project_id)

    await _set_ingest_status(
        redis,
        project_id,
        status="running",
        message="Ingestion started.",
        queued_at=queued_at,
        started_at=started_at,
        updated_at=started_at,
        job_id=job_id,
    )

    try:
        outcome = await _run_ingestion(ctx, project_id, oldest_timestamp=None)
    except asyncio.CancelledError:
        finished_at = datetime.now(UTC).isoformat()
        await asyncio.shield(
            _set_ingest_status(
                redis,
                project_id,
                status="failed",
                message="Ingestion cancelled.",
                queued_at=queued_at,
                started_at=started_at,
                finished_at=finished_at,
                updated_at=finished_at,
                job_id=job_id,
            )
        )
        raise
    except Exception as exc:
        finished_at = datetime.now(UTC).isoformat()
        await asyncio.shield(
            _set_ingest_status(
                redis,
                project_id,
                status="failed",
                message=f"Ingestion failed: {type(exc).__name__}",
                queued_at=queued_at,
                started_at=started_at,
                finished_at=finished_at,
                updated_at=finished_at,
                job_id=job_id,
            )
        )
        raise

    finished_at = datetime.now(UTC).isoformat()
    await _set_ingest_status(
        redis,
        project_id,
        status=str(outcome.get("status", "unknown")),
        message=str(outcome.get("message", "")),
        queued_at=queued_at,
        started_at=started_at,
        finished_at=finished_at,
        updated_at=finished_at,
        source_counts=outcome.get("source_counts") or {},
        source_diagnostics=outcome.get("source_diagnostics") or {},
        fulltext_enqueued=int(outcome.get("fulltext_enqueued") or 0),
        total_chunks=outcome.get("total_chunks"),
        job_id=job_id,
    )


async def refresh_project(ctx: dict, project_id: str, oldest_timestamp: int) -> None:
    """Incremental refresh of project KB since oldest_timestamp."""
    redis = await get_redis()
    started_at = datetime.now(UTC).isoformat()
    await _set_ingest_status(
        redis,
        project_id,
        status="running",
        message="Incremental refresh started.",
        started_at=started_at,
        updated_at=started_at,
    )

    try:
        outcome = await _run_ingestion(ctx, project_id, oldest_timestamp=oldest_timestamp)
        # Flush project caches to reflect new data
        await invalidate_project_caches(project_id)
    except Exception as exc:
        finished_at = datetime.now(UTC).isoformat()
        await _set_ingest_status(
            redis,
            project_id,
            status="failed",
            message=f"Refresh failed: {type(exc).__name__}",
            started_at=started_at,
            finished_at=finished_at,
            updated_at=finished_at,
        )
        raise

    finished_at = datetime.now(UTC).isoformat()
    await _set_ingest_status(
        redis,
        project_id,
        status=str(outcome.get("status", "unknown")),
        message=str(outcome.get("message", "")),
        started_at=started_at,
        finished_at=finished_at,
        updated_at=finished_at,
        source_counts=outcome.get("source_counts") or {},
        source_diagnostics=outcome.get("source_diagnostics") or {},
        fulltext_enqueued=int(outcome.get("fulltext_enqueued") or 0),
        total_chunks=outcome.get("total_chunks"),
    )


async def _run_ingestion(
    ctx: dict,
    project_id: str,
    oldest_timestamp: int | None,
) -> dict:
    """Core ingestion logic shared by ingest_project and refresh_project."""
    pool = await get_db_pool()
    redis = await get_redis()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, user_id::text, title, description, structured_intent,
                   tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled,
                   papers_enabled, patents_enabled, perigon_enabled, tavily_enabled, exa_enabled
            FROM projects
            WHERE id = $1::uuid
            """,
            project_id,
        )

    if not row:
        return {"status": "failed", "message": "Project not found."}

    project = dict(row)
    raw_intent = project.get("structured_intent") or {}
    if isinstance(raw_intent, str):
        try:
            raw_intent = json.loads(raw_intent)
        except (json.JSONDecodeError, ValueError):
            raw_intent = {}
    intent = dict(raw_intent)
    user_id = project["user_id"]
    project_title = str(project.get("title") or "")
    project_description = str(project.get("description") or "")

    social_filter = str((intent.get("search_filters") or {}).get("social") or "").strip()
    if not social_filter:
        social_filter = _query_for_social(
            intent,
            project_title=project_title,
            project_description=project_description,
        )

    tiktok_enabled = bool(project.get("tiktok_enabled"))
    instagram_enabled = bool(project.get("instagram_enabled"))
    youtube_enabled = bool(project.get("youtube_enabled"))
    reddit_enabled = bool(project.get("reddit_enabled"))
    x_enabled = bool(project.get("x_enabled"))
    papers_enabled = bool(project.get("papers_enabled"))
    patents_enabled = bool(project.get("patents_enabled"))
    perigon_enabled = bool(project.get("perigon_enabled"))
    tavily_enabled = bool(project.get("tavily_enabled"))
    exa_enabled = bool(project.get("exa_enabled"))

    strict_social_terms = sorted(
        _project_anchor_terms(
            intent,
            social_filter=social_filter,
            project_title=project_title,
            project_description=project_description,
        )
    )

    intent_text = json.dumps(intent)
    try:
        # Win #5: Pre-compute intent embedding once to reuse across stages (relevance + social)
        intent_embeddings = await embed_texts([intent_text])
        intent_embedding = intent_embeddings[0] if intent_embeddings else None
    except Exception:
        logger.exception("ingestion.intent_embedding_failed")
        intent_embedding = None

    source_types = [
        "social_tiktok",
        "social_instagram",
        "social_youtube",
        "social_reddit",
        "social_x",
        "paper_openalex",
        "paper_semantic_scholar",
        "paper_pubmed",
        "paper_arxiv",
        "patent_patentsview",
        "patent_lens",
        "patent_web_fallback",
        "news_perigon",
        "search_tavily",
        "search_exa",
    ]

    coroutines = [
        _run_source_with_timeout(
            "social_tiktok",
            _ingest_tiktok(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_SOCIAL_SOURCE_TIMEOUT,
        )
        if tiktok_enabled
        else _noop(),
        _run_source_with_timeout(
            "social_instagram",
            _ingest_instagram(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                oldest_timestamp=oldest_timestamp,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_SOCIAL_SOURCE_TIMEOUT,
        )
        if instagram_enabled
        else _noop(),
        _run_source_with_timeout(
            "social_youtube",
            _ingest_youtube(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_SOCIAL_SOURCE_TIMEOUT,
        )
        if youtube_enabled
        else _noop(),
        _run_source_with_timeout(
            "social_reddit",
            _ingest_reddit(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_SOCIAL_SOURCE_TIMEOUT,
        )
        if reddit_enabled
        else _noop(),
        _run_source_with_timeout(
            "social_x",
            _ingest_x(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_SOCIAL_SOURCE_TIMEOUT,
        )
        if x_enabled
        else _noop(),
        _run_source_with_timeout(
            "paper_openalex",
            _ingest_papers_openalex(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
            ),
            timeout_seconds=settings.INGEST_PAPERS_TIMEOUT,
        )
        if papers_enabled
        else _noop(),
        _run_source_with_timeout(
            "paper_semantic_scholar",
            _ingest_papers_semantic_scholar(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_PAPERS_TIMEOUT,
        )
        if papers_enabled
        else _noop(),
        _run_source_with_timeout(
            "paper_pubmed",
            _ingest_papers_pubmed(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_PAPERS_TIMEOUT,
        )
        if papers_enabled
        else _noop(),
        _run_source_with_timeout(
            "paper_arxiv",
            _ingest_papers_arxiv(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_PAPERS_TIMEOUT,
        )
        if papers_enabled
        else _noop(),
        _run_source_with_timeout(
            "patent_patentsview",
            _ingest_patents_patentsview(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_PATENTS_TIMEOUT,
        )
        if patents_enabled
        else _noop(),
        _run_source_with_timeout(
            "patent_lens",
            _ingest_patents_lens(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                project_title=project_title,
                project_description=project_description,
                redis=redis,
            ),
            timeout_seconds=settings.INGEST_PATENTS_TIMEOUT,
        )
        if patents_enabled
        else _noop(),
        _run_source_with_timeout(
            "patent_web_fallback",
            _ingest_patents_web_fallback(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                project_title=project_title,
                project_description=project_description,
            ),
            timeout_seconds=settings.INGEST_PATENTS_TIMEOUT,
        )
        if patents_enabled
        else _noop(),
        _run_source_with_timeout(
            "news_perigon",
            _ingest_news(project_id=project_id, user_id=user_id, intent=intent, redis=redis),
            timeout_seconds=settings.INGEST_NEWS_TIMEOUT,
        )
        if perigon_enabled
        else _noop(),
        _run_source_with_timeout(
            "search_tavily",
            _ingest_web_tavily(project_id=project_id, user_id=user_id, intent=intent, redis=redis),
        )
        if tavily_enabled
        else _noop(),
        _run_source_with_timeout(
            "search_exa",
            _ingest_web_exa(project_id=project_id, user_id=user_id, intent=intent, redis=redis),
        )
        if exa_enabled
        else _noop(),
    ]

    async def _run_indexed(index: int, operation):
        return index, await operation

    mutable_results: list[list] = [[] for _ in source_types]
    tasks = [asyncio.create_task(_run_indexed(idx, coro)) for idx, coro in enumerate(coroutines)]

    intent_text = json.dumps(intent)

    # Track metadata to avoid OOM from huge RawDocument arrays
    source_counts: dict[str, int] = {}
    source_diagnostics: dict[str, dict] = {}
    seen_social_ids: set[tuple[str, str]] = set()
    expansion_seeds: dict[str, set[str]] = {
        "social_instagram": set(),
        "social_tiktok": set(),
        "social_youtube": set(),
        "social_reddit": set(),
        "social_x": set(),
    }

    any_source_completed = False
    inserted = 0
    fulltext_enqueued = 0
    enabled_sources_total = sum(
        1
        for enabled in [
            tiktok_enabled,
            instagram_enabled,
            youtube_enabled,
            reddit_enabled,
            x_enabled,
            papers_enabled,
            papers_enabled,
            papers_enabled,
            papers_enabled,
            patents_enabled,
            patents_enabled,
            patents_enabled,
            perigon_enabled,
            tavily_enabled,
            exa_enabled,
        ]
        if enabled
    )

    # 1. Non-Social Processing — process tasks as they complete
    for completed_sources, task in enumerate(asyncio.as_completed(tasks), start=1):
        idx, result = await task
        mutable_results[idx] = result if isinstance(result, list) else []
        source_key = source_types[idx]

        if source_key.startswith("social_"):
            source = source_key
        elif source_key.startswith("paper_"):
            source = "paper"
        elif source_key.startswith("patent_"):
            source = "patent"
        elif source_key.startswith("news_"):
            source = "news"
        elif source_key.startswith("search_"):
            source = "search"
        else:
            source = source_key

        if completed_sources % 3 == 0:
            async with pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT 1 FROM projects WHERE id = $1::uuid", project_id
                )
                if not exists:
                    logger.info("ingestion.project_deleted", project_id=project_id)
                    break

        # Process all sources (including social) immediately as they complete
        # Social sources no longer delayed - processed in first loop

        kept_docs, diagnostics, source_completed = await _prepare_kept_documents_for_source(
            result=result,
            source=source,
            source_enabled=True,
            oldest_timestamp=oldest_timestamp,
            strict_social_terms=strict_social_terms,
            intent_text=intent_text,
            redis=redis,
            expansion_meta={},
        )
        source_diagnostics[source_key] = diagnostics
        any_source_completed = any_source_completed or source_completed

        if kept_docs:
            inserted += await ingest_documents(kept_docs)
            fulltext_enqueued += await _schedule_fulltext_enrichment(ctx, pool, kept_docs)
            source_counts[source] = source_counts.get(source, 0) + len(kept_docs)

            # When expansion is enabled, track social doc IDs inserted in this
            # first pass so Section 2 doesn't double-insert the same documents.
            if settings.INGEST_SOCIAL_EXPANSION_ENABLED and source_key.startswith("social_"):
                for d in kept_docs:
                    seen_social_ids.add((d.source, str(d.source_id)))

        await _set_ingest_status(
            redis,
            project_id,
            status="running",
            message=f"Ingestion in progress ({completed_sources}/{enabled_sources_total} completed)",
            source_counts=source_counts,
            source_diagnostics=source_diagnostics,
            fulltext_enqueued=fulltext_enqueued,
        )

    # 2. Social Processing & seed extraction (only if expansion is enabled)
    # Since expansion is disabled, this loop is skipped for better performance
    if settings.INGEST_SOCIAL_EXPANSION_ENABLED:
        any_social_completed = False

        for idx, source_key in enumerate(source_types):
            if not source_key.startswith("social_"):
                continue

            # BUG FIX: This block must be INSIDE the for loop (indented one more level)
            # Previously it sat at the same indent level as `for`, meaning it only ran
            # once after the loop exited — processing only the last social source.
            source = source_key
            result = mutable_results[idx]

            kept, diagnostics, source_completed = await _prepare_kept_documents_for_source(
                result=result,
                source=source,
                source_enabled=True,
                oldest_timestamp=oldest_timestamp,
                strict_social_terms=strict_social_terms,
                intent_text=intent_text,
                redis=redis,
                expansion_meta={},
                intent_embedding=intent_embedding,
            )
            source_diagnostics[source] = diagnostics
            any_social_completed = any_social_completed or source_completed

            if kept:
                # Extract seeds from ALL kept docs — including ones already inserted
                # in Section 1. Seed extraction must not be gated on dedup, otherwise
                # expansion seeds are silently lost when seen_social_ids is pre-populated.
                seeds = _extract_expansion_links(kept)
                for k, v in seeds.items():
                    expansion_seeds[k].update(v)

                # Only insert docs that weren't already inserted in Section 1
                unique_kept = [
                    d for d in kept if (d.source, str(d.source_id)) not in seen_social_ids
                ]
                for d in unique_kept:
                    seen_social_ids.add((d.source, str(d.source_id)))

                if unique_kept:
                    inserted += await ingest_documents(unique_kept)
                    source_counts[source] = source_counts.get(source, 0) + len(unique_kept)

    # 3. Social Expansion
    if settings.INGEST_SOCIAL_EXPANSION_ENABLED and any(expansion_seeds.values()):
        expansion_coros = []

        if tiktok_enabled and expansion_seeds["social_tiktok"]:
            for q in list(expansion_seeds["social_tiktok"])[:5]:
                expansion_coros.append(
                    _run_source_with_timeout(
                        "social_tiktok_exp",
                        _ingest_tiktok(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=q,
                            expansion_mode=True,
                            redis=redis,
                        ),
                    )
                )
        if instagram_enabled and expansion_seeds["social_instagram"]:
            for q in list(expansion_seeds["social_instagram"])[:5]:
                expansion_coros.append(
                    _run_source_with_timeout(
                        "social_instagram_exp",
                        _ingest_instagram(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=q,
                            expansion_mode=True,
                            oldest_timestamp=oldest_timestamp,
                            redis=redis,
                        ),
                    )
                )
        if youtube_enabled and expansion_seeds["social_youtube"]:
            for q in list(expansion_seeds["social_youtube"])[:5]:
                expansion_coros.append(
                    _run_source_with_timeout(
                        "social_youtube_exp",
                        _ingest_youtube(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=q,
                            expansion_mode=True,
                            redis=redis,
                        ),
                    )
                )
        if reddit_enabled and expansion_seeds["social_reddit"]:
            for q in list(expansion_seeds["social_reddit"])[:5]:
                expansion_coros.append(
                    _run_source_with_timeout(
                        "social_reddit_exp",
                        _ingest_reddit(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=q,
                            expansion_mode=True,
                            redis=redis,
                        ),
                    )
                )
        if x_enabled and expansion_seeds["social_x"]:
            for q in list(expansion_seeds["social_x"])[:5]:
                expansion_coros.append(
                    _run_source_with_timeout(
                        "social_x_exp",
                        _ingest_x(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=q,
                            expansion_mode=True,
                            redis=redis,
                        ),
                    )
                )

        if expansion_coros:
            logger.info("ingestion.social_expansion.start", task_count=len(expansion_coros))
            exp_results = await asyncio.gather(*expansion_coros, return_exceptions=True)

            for _result in exp_results:
                kept_exp, _, _ = await _prepare_kept_documents_for_source(
                    result=_result,
                    source="social_expansion",
                    source_enabled=True,
                    oldest_timestamp=oldest_timestamp,
                    strict_social_terms=strict_social_terms,
                    intent_text=intent_text,
                    redis=redis,
                    expansion_meta={},
                    intent_embedding=intent_embedding,
                )
                if kept_exp:
                    unique_exp = []
                    for d in kept_exp:
                        # social_expansion resolves to actual source inside the document
                        key = (d.source, str(d.source_id))
                        if key not in seen_social_ids:
                            seen_social_ids.add(key)
                            unique_exp.append(d)

                    if unique_exp:
                        inserted += await ingest_documents(unique_exp)
                        # We accumulate social_expansion counts into the parent real source bucket
                        for d in unique_exp:
                            source_counts[d.source] = source_counts.get(d.source, 0) + 1

    finished_at = datetime.now(UTC).isoformat()

    # Query the real cumulative chunk total from the DB — `inserted` is only this
    # run's new chunks. On refresh, overwriting with `inserted` would undercount.
    real_total_chunks = inserted  # fallback if DB query fails
    try:
        async with pool.acquire() as conn:
            real_total_chunks = (
                await conn.fetchval(
                    "SELECT COUNT(*)::int FROM knowledge_chunks WHERE project_id = $1::uuid",
                    project_id,
                )
                or 0
            )
    except Exception:
        logger.warning("ingestion.total_chunk_count_query_failed", project_id=project_id)

    outcome = {
        "status": "ready" if real_total_chunks > 0 else "empty",
        "message": f"Ingestion complete. {inserted} new chunks added ({real_total_chunks} total)."
        if inserted > 0
        else "No new chunks found."
        if real_total_chunks == 0
        else f"No new chunks added. {real_total_chunks} chunks already in KB.",
        "finished_at": finished_at,
        "source_counts": source_counts,
        "source_diagnostics": source_diagnostics,
        "fulltext_enqueued": fulltext_enqueued,
        "total_chunks": real_total_chunks,
    }

    # Update DB stats with the accurate cumulative total
    try:
        await project_repo.update_project_kb_stats(
            pool, project_id, real_total_chunks, datetime.now(UTC)
        )
    except Exception:
        logger.warning("ingestion.db_stats_update_failed", project_id=project_id)

    await _set_ingest_status(
        redis,
        project_id,
        status=outcome["status"],
        message=outcome["message"],
        finished_at=finished_at,
        source_counts=outcome["source_counts"],
        source_diagnostics=source_diagnostics,
        fulltext_enqueued=fulltext_enqueued,
        total_chunks=real_total_chunks,
    )
    return outcome
