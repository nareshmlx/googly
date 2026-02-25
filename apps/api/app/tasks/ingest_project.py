"""ingest_project ARQ task — initial KB population from enabled sources after project creation.

Called immediately after project creation via ARQ enqueue.
Also re-used by refresh_project for subsequent refreshes.

Supports independently toggleable ingestion sources per project:
  - Social: Instagram, TikTok
  - Papers: OpenAlex, Semantic Scholar, PubMed, arXiv
  - Patents: PatentsView, Lens
  - Discovery: Perigon news, Tavily web, Exa web

Partial success is intentional: one source failing never aborts the others.
"""

import asyncio
import json
import math
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from openai import AsyncOpenAI

from app.core.cache_version import bump_project_cache_version
from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.db import get_db_pool
from app.core.query_policy import build_query_policy, build_source_query, lexical_entity_coverage
from app.core.redis import get_redis
from app.kb.embedder import embed_texts
from app.kb.ingester import RawDocument, ingest_documents
from app.repositories import project as project_repo
from app.tools.news_perigon import search_perigon
from app.tools.papers_arxiv import search_arxiv
from app.tools.papers_openalex import fetch_papers
from app.tools.papers_pubmed import search_pubmed
from app.tools.papers_semantic_scholar import search_semantic_scholar
from app.tools.patents_lens import search_lens
from app.tools.patents_patentsview import search_patentsview
from app.tools.search_exa import search_exa
from app.tools.search_tavily import search_tavily
from app.tools.social_instagram import (
    instagram_hashtag_posts,
    instagram_search,
    instagram_user_reels,
)
from app.tools.social_tiktok import fetch_tiktok_posts

logger = structlog.get_logger(__name__)

RELEVANCE_STAGE2_BATCH_SIZE = 15
RELEVANCE_MIN_STAGE1_KEEP_BY_SOURCE: dict[str, int] = {
    "paper": 12,
    "patent": 8,
    "news": 8,
    "search": 8,
}
RELEVANCE_MIN_STAGE2_KEEP_BY_SOURCE: dict[str, int] = {
    "paper": 6,
    "patent": 4,
}


async def _noop() -> list[RawDocument]:
    """Return an empty source result for disabled gather slots."""
    return []


async def _run_source_with_timeout(source: str, coro) -> list[RawDocument]:
    """Run one source ingest with timeout and fail-open behavior."""
    try:
        return await asyncio.wait_for(coro, timeout=settings.INGEST_SOURCE_TIMEOUT)
    except TimeoutError:
        logger.warning(
            "ingest_tool.timeout",
            source=source,
            timeout_seconds=settings.INGEST_SOURCE_TIMEOUT,
        )
        return []
    except Exception as exc:
        logger.warning("ingest_tool.failed", source=source, error=str(exc))
        return []


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
    total_chunks: int | None = None,
) -> None:
    """Persist ingest lifecycle status for API/UI visibility."""
    payload = {
        "project_id": project_id,
        "status": status,
        "message": message,
        "queued_at": queued_at,
        "started_at": started_at,
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
        "finished_at": finished_at,
        "source_counts": source_counts or {},
        "total_chunks": total_chunks,
    }
    try:
        key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)
        await redis.setex(key, RedisTTL.PROJECT_INGEST_STATUS.value, json.dumps(payload))
    except Exception:
        logger.warning("ingest_project.status_set_failed", project_id=project_id, status=status)


def _is_document_new_enough(doc: RawDocument, oldest_timestamp: int | None) -> bool:
    """Return True when a document is newer than the refresh watermark.

    If no watermark is provided or publish metadata is unavailable/invalid,
    the document is kept (fail-open).
    """
    if not isinstance(oldest_timestamp, int | float) or oldest_timestamp <= 0:
        return True

    metadata = doc.metadata or {}
    published_raw = (
        metadata.get("published_at")
        or metadata.get("published")
        or metadata.get("published_date")
        or metadata.get("date")
        or metadata.get("year")
        or metadata.get("publication_year")
        or metadata.get("timestamp")
    )
    if published_raw in (None, ""):
        return True

    try:
        if isinstance(published_raw, int | float):
            published_dt = datetime.fromtimestamp(float(published_raw), tz=UTC)
        elif isinstance(published_raw, str) and published_raw.isdigit() and len(published_raw) == 4:
            published_dt = datetime(int(published_raw), 1, 1, tzinfo=UTC)
        else:
            normalized = str(published_raw).replace("Z", "+00:00")
            published_dt = datetime.fromisoformat(normalized)
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=UTC)
        return published_dt.timestamp() >= float(oldest_timestamp)
    except Exception:
        return True


def _relevance_item_text(item: dict) -> str:
    """Build a stable short text for relevance embedding comparison."""
    title = str(item.get("title") or "").strip()
    body = str(item.get("abstract") or item.get("content") or item.get("claims") or "").strip()
    text = f"{title} {body[:400]}".strip()
    return text or "untitled"


async def _filter_relevance(
    items: list[dict],
    intent_text: str,
    source: str,
    redis,
    must_match_terms: list[str] | None = None,
) -> list[dict]:
    """Apply two-stage relevance filtering with fail-open behavior."""
    if not items:
        return items

    if source in {"social_tiktok", "social_instagram"}:
        logger.info("relevance_filter.skipped_social", source=source, total=len(items))
        return items

    filtered = items
    try:
        filtered = await _filter_stage1_embedding(filtered, intent_text, redis, source=source)
    except Exception:
        logger.exception("relevance_filter.stage1_failed", source=source, total=len(items))
        filtered = items

    if source in {"paper", "patent"} and filtered:
        try:
            stage2_filtered = await _filter_stage2_llm(
                filtered,
                intent_text,
                source,
                must_match_terms=must_match_terms,
            )
            min_keep = RELEVANCE_MIN_STAGE2_KEEP_BY_SOURCE.get(source, 0)
            if min_keep > 0 and len(stage2_filtered) < min_keep:
                fallback_count = min(min_keep, len(filtered))
                logger.info(
                    "relevance_filter.stage2_min_fallback",
                    source=source,
                    llm_kept=len(stage2_filtered),
                    min_keep=min_keep,
                    fallback_count=fallback_count,
                )
                return filtered[:fallback_count]
            filtered = stage2_filtered
        except Exception:
            logger.exception("relevance_filter.stage2_failed", source=source, total=len(filtered))

    return filtered


async def _filter_stage1_embedding(
    items: list[dict],
    intent_text: str,
    redis,
    source: str | None = None,
) -> list[dict]:
    """Drop items in the bottom 40th percentile of intent cosine similarity."""
    if not items:
        return items

    _ = redis
    intent_embeddings = await embed_texts([intent_text])
    if not intent_embeddings:
        return items
    intent_arr = np.array(intent_embeddings[0], dtype=float)

    texts = [_relevance_item_text(item) for item in items]
    item_embeddings = await embed_texts(texts)
    if len(item_embeddings) != len(items):
        logger.warning(
            "relevance_filter.stage1_mismatch",
            expected=len(items),
            actual=len(item_embeddings),
        )
        return items

    intent_norm = float(np.linalg.norm(intent_arr))
    scores: list[float] = []
    for embedding in item_embeddings:
        item_arr = np.array(embedding, dtype=float)
        denom = intent_norm * float(np.linalg.norm(item_arr))
        if denom <= 0:
            scores.append(0.0)
            continue
        score = float(np.dot(intent_arr, item_arr) / denom)
        scores.append(score)

    threshold = float(np.percentile(np.array(scores, dtype=float), 40))
    scored_items = list(zip(items, scores, strict=False))
    scored_items.sort(key=lambda pair: pair[1], reverse=True)

    filtered_pairs = [pair for pair in scored_items if pair[1] >= threshold]

    min_keep = RELEVANCE_MIN_STAGE1_KEEP_BY_SOURCE.get(source or "", 0)
    if min_keep > 0 and len(filtered_pairs) < min_keep:
        filtered_pairs = scored_items[: min(min_keep, len(scored_items))]

    filtered = [item for item, score in filtered_pairs]
    for item, score in filtered_pairs:
        item["_stage1_similarity"] = round(float(score), 6)

    logger.info(
        "relevance_filter.stage1",
        source=source,
        kept=len(filtered),
        total=len(items),
        threshold=round(threshold, 4),
    )
    return filtered


async def _filter_stage2_llm(
    items: list[dict],
    intent_text: str,
    source: str,
    must_match_terms: list[str] | None = None,
) -> list[dict]:
    """Use batched GPT-4o-mini relevance judging for papers and patents."""
    if not items:
        return items

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    kept_items: list[dict] = []

    for batch_start in range(0, len(items), RELEVANCE_STAGE2_BATCH_SIZE):
        batch = items[batch_start : batch_start + RELEVANCE_STAGE2_BATCH_SIZE]
        numbered: list[str] = []
        for index, item in enumerate(batch):
            title = str(item.get("title") or "Unknown title")
            if source == "paper":
                body_label = "Abstract"
                body_text = str(item.get("abstract") or item.get("content") or "")
            else:
                body_label = "Claims"
                body_text = str(
                    item.get("claims") or item.get("abstract") or item.get("content") or ""
                )
            numbered.append(f"{index}. Title: {title}\n{body_label}: {body_text[:500]}")

        must_terms = [str(term).strip() for term in (must_match_terms or []) if str(term).strip()]
        must_terms_line = ""
        if must_terms:
            must_terms_line = (
                "Specific query must-match terms: "
                + ", ".join(must_terms[:5])
                + ". Mark an item relevant=1 only if it explicitly contains at least one of these terms.\n\n"
            )

        prompt = (
            f"Project intent:\n{intent_text[:1200]}\n\n"
            f"{must_terms_line}"
            "For each item, return strict JSON in this shape: "
            '{"items": [{"id": <number>, "relevant": 0 or 1}]}. '
            "Mark relevant=1 only when clearly aligned with the project intent.\n\n"
            + "\n\n".join(numbered)
        )

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict relevance classifier. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            entries = parsed.get("items", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(entries, list):
                logger.warning(
                    "relevance_filter.stage2_invalid_shape",
                    source=source,
                    batch_start=batch_start,
                )
                kept_items.extend(batch)
                continue

            relevant_ids: set[int] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    if int(entry.get("relevant", 0)) != 1:
                        continue
                    entry_id = int(entry.get("id", -1))
                except (TypeError, ValueError):
                    continue
                if entry_id >= 0:
                    relevant_ids.add(entry_id)
            for index, item in enumerate(batch):
                if index not in relevant_ids:
                    continue
                if must_terms:
                    text = f"{item.get('title', '')} {item.get('abstract', '')} {item.get('content', '')}"
                    coverage = lexical_entity_coverage(text, must_terms)
                    if coverage < settings.QUERY_ENTITY_MATCH_THRESHOLD:
                        continue
                kept_items.append(item)
        except Exception:
            logger.exception(
                "relevance_filter.stage2_batch_failed",
                source=source,
                batch_start=batch_start,
                batch_size=len(batch),
            )
            kept_items.extend(batch)

    logger.info("relevance_filter.stage2", source=source, kept=len(kept_items), total=len(items))
    return kept_items


async def _invalidate_project_caches(project_id: str) -> None:
    """
    Invalidate all caches for a project after KB update.

    Bumps project cache version so semantic + KB hot cache keys rotate immediately,
    then clears project-scoped search cache keys via SCAN.

    Cache invalidation failure is logged but does not crash the ingestion
    task — data consistency is maintained even if caches remain stale.
    """
    try:
        redis = await get_redis()
        new_version = await bump_project_cache_version(redis, project_id)
        patterns = [
            f"search:cache:{project_id}:*",
        ]

        deleted_count = 0
        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis.delete(*keys)
                    deleted_count += len(keys)
                if cursor == 0:
                    break

        logger.info(
            "cache_invalidation.complete",
            project_id=project_id,
            cache_version=new_version,
            keys_deleted=deleted_count,
        )
    except Exception:
        logger.exception(
            "cache_invalidation.failed",
            project_id=project_id,
        )


ACCOUNT_RELEVANCE_WEIGHT = 6.0
ACCOUNT_BRAND_WEIGHT = 5.0
ACCOUNT_CREDIBILITY_WEIGHT = 2.0
ACCOUNT_REACH_WEIGHT = 3.0
ACCOUNT_SCORE_WEIGHT = 1.0
REEL_RELEVANCE_WEIGHT = 6.0
REEL_BRAND_WEIGHT = 5.0
REEL_ENGAGEMENT_WEIGHT = 8.0
REEL_RECENCY_WEIGHT = 2.0
REEL_RECENCY_DAYS = 30.0

_BEAUTY_SCOPE_TERMS: set[str] = {
    "beauty",
    "cosmetic",
    "cosmetics",
    "skincare",
    "skin",
    "makeup",
    "fragrance",
    "perfume",
    "haircare",
    "moisturizer",
    "sunscreen",
    "balm",
    "lipbalm",
    "lip",
    "lipstick",
    "lipgloss",
    "concealer",
    "foundation",
    "serum",
    "cleanser",
    "toner",
    "exfoliant",
    "acne",
    "pigmentation",
    "bodycare",
    "retinol",
    "niacinamide",
    "hyaluronic",
    "peptides",
    "ceramide",
    "aha",
    "bha",
}

_GENERIC_RELEVANCE_TERMS: set[str] = {
    "trend",
    "trends",
    "research",
    "analysis",
    "market",
    "industry",
    "product",
    "products",
    "news",
    "viral",
}


def _as_int(value: object) -> int:
    """Convert mixed numeric payload values (int/float/str) to int safely."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return 0
        try:
            return int(float(cleaned))
        except ValueError:
            return 0
    return 0


def _tokenize(text: str) -> set[str]:
    """Lowercase token set for keyword relevance checks."""
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 3}


def _keyword_tokens(text: str) -> set[str]:
    """Tokenize keywords and add a compact alphanumeric form for brand-like terms."""
    tokens = _tokenize(text)
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(compact) >= 3:
        tokens.add(compact)
    return tokens


def _contains_beauty_signal(text: str) -> bool:
    """Return True if text includes at least one beauty-domain token."""
    return len(_tokenize(text) & _BEAUTY_SCOPE_TERMS) > 0


def _filter_relevance_terms(terms: set[str]) -> set[str]:
    """Remove low-signal generic tokens while preserving beauty-domain terms."""
    filtered = {t for t in terms if t and t not in _GENERIC_RELEVANCE_TERMS}
    return filtered


def _build_relevance_terms(intent: dict, social_filter: str) -> set[str]:
    """
    Build keyword terms used to rank Instagram accounts and reels.

    Terms come from extracted keywords, instagram search filter, and social hashtags.
    """
    terms: set[str] = set()
    keywords: list[str] = intent.get("keywords") or []
    for kw in keywords:
        terms.update(_keyword_tokens(str(kw)))

    search_filters = intent.get("search_filters") or {}
    terms.update(_keyword_tokens(str(search_filters.get("instagram") or "")))
    terms.update(_keyword_tokens(str(social_filter or "")))
    terms = _filter_relevance_terms(terms)
    if not terms:
        terms = {"beauty", "cosmetics", "skincare", "makeup", "fragrance"}
    return terms


def _build_brand_terms(intent: dict) -> set[str]:
    """
    Extract non-generic brand/product tokens from intent keywords.

    These are used as a boost signal so projects like P&G/Olay/SK-II do not
    get dominated by generic 'beauty' accounts.
    """
    keywords: list[str] = intent.get("keywords") or []
    terms: set[str] = set()
    for kw in keywords:
        for tok in _keyword_tokens(str(kw)):
            if tok in _GENERIC_RELEVANCE_TERMS:
                continue
            if tok in _BEAUTY_SCOPE_TERMS:
                continue
            terms.add(tok)
    return terms


def _match_count(text: str, terms: set[str]) -> int:
    """Count distinct relevance-term matches in text."""
    if not text or not terms:
        return 0
    tokens = _tokenize(text)
    return len(tokens & terms)


def _clean_instagram_keyword_query(raw: str) -> str:
    """
    Convert a raw keyword into a safe short Instagram search query.

    Keeps only letters/numbers/spaces, collapses whitespace, and limits to
    max 3 words to match the Ensemble Instagram user-search behavior.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    words = cleaned.split()[:3]
    if not words:
        return ""
    return " ".join(words)


def _extract_hashtags(text: str) -> list[str]:
    """Extract cleaned hashtag terms from a social filter string."""
    if not text:
        return []
    tags = re.findall(r"#([a-zA-Z0-9_]+)", text)
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = _clean_instagram_keyword_query(tag.replace("_", " "))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _instagram_handle_candidates(social_filter: str, intent: dict) -> list[str]:
    """Build candidate Instagram handle queries for fallback account lookup."""
    candidates: list[str] = []
    seen: set[str] = set()

    raw_terms: list[str] = []
    if social_filter:
        raw_terms.append(str(social_filter))
        raw_terms.extend(str(token) for token in str(social_filter).split())
    raw_terms.extend(str(term) for term in (intent.get("keywords") or [])[:6])

    for raw in raw_terms:
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        cleaned = cleaned.lstrip("#@")
        cleaned = re.sub(r"[^a-zA-Z0-9_.\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cleaned)
        if len(candidates) >= 10:
            break

    return candidates


async def _fetch_instagram_reels_from_candidates(
    *,
    project_id: str,
    candidate_queries: list[str],
    oldest_timestamp: int | None,
) -> list[dict]:
    """Fetch reels by searching candidate handles and resolving account IDs."""
    reels_out: list[dict] = []
    seen_users: set[int] = set()

    for query in candidate_queries:
        try:
            search_results = await instagram_search(query)
        except Exception:
            logger.exception(
                "ingest_instagram.fallback_search_failed",
                project_id=project_id,
                query=query,
            )
            continue
        if not search_results:
            continue

        exact = next(
            (
                row
                for row in search_results
                if str(row.get("username") or "").strip().lower() == query.lower()
            ),
            None,
        )
        candidate = exact or search_results[0]
        raw_uid = candidate.get("pk") or candidate.get("id")
        if not raw_uid:
            continue
        try:
            uid = int(str(raw_uid))
        except (TypeError, ValueError):
            continue
        if uid in seen_users:
            continue
        seen_users.add(uid)

        username = str(candidate.get("username") or query).strip()
        try:
            reels = await instagram_user_reels(
                uid,
                depth=1,
                oldest_timestamp=oldest_timestamp,
            )
        except Exception:
            logger.exception(
                "ingest_instagram.fallback_reels_failed",
                project_id=project_id,
                query=query,
                uid=uid,
            )
            continue

        for reel in reels or []:
            if not reel.get("username"):
                reel["username"] = username
            reels_out.append(reel)

        if len(reels_out) >= max(20, settings.INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT):
            break

    logger.info(
        "ingest_instagram.fallback_done",
        project_id=project_id,
        candidate_queries=len(candidate_queries),
        reel_count=len(reels_out),
    )
    return reels_out


async def ingest_project(ctx: dict, project_id: str) -> None:
    """
    Populate a project's KB from all enabled sources based on its structured_intent.

    Flow:
    1. Fetch project row including per-source toggle columns
    2. Validate social_filter from structured_intent.search_filters.social
    3. Fan out to each enabled source (Instagram, TikTok, OpenAlex) independently
    4. Build RawDocument list from all collected content
    5. ingest_documents() → chunk, embed, upsert to knowledge_chunks
    6. Update kb_chunk_count and last_refreshed_at

    oldest_timestamp=None means fetch all available content (initial run).
    For refresh runs, pass oldest_timestamp via refresh_project instead.
    """
    redis = await get_redis()
    started_at = datetime.now(UTC).isoformat()
    queued_at = None
    try:
        key = RedisKeys.PROJECT_INGEST_STATUS.format(project_id=project_id)
        raw = await redis.get(key)
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                queued_at = parsed.get("queued_at")
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
    )
    try:
        outcome = await _run_ingestion(ctx, project_id, oldest_timestamp=None)
    except Exception as exc:
        finished_at = datetime.now(UTC).isoformat()
        await _set_ingest_status(
            redis,
            project_id,
            status="failed",
            message=f"Ingestion failed: {type(exc).__name__}",
            queued_at=queued_at,
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
        queued_at=queued_at,
        started_at=started_at,
        finished_at=finished_at,
        updated_at=finished_at,
        source_counts=outcome.get("source_counts") or {},
        total_chunks=outcome.get("total_chunks"),
    )


async def _run_ingestion(
    ctx: dict,
    project_id: str,
    oldest_timestamp: int | None,
) -> dict:
    """
    Core ingestion logic shared by ingest_project and refresh_project.

    Fans out to all enabled sources independently.  A source returning an empty
    list is treated as a warning, not a fatal error — other sources continue.
    Separated so refresh_project can pass oldest_timestamp without duplicating
    the full ingestion flow.
    """
    pool = await get_db_pool()
    redis = await get_redis()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, user_id::text, title, description, structured_intent, last_refreshed_at,
                   tiktok_enabled, instagram_enabled, papers_enabled,
                   patents_enabled, perigon_enabled, tavily_enabled, exa_enabled
            FROM projects
            WHERE id = $1::uuid
            """,
            project_id,
        )

    if not row:
        logger.warning("ingest_project.project_not_found", project_id=project_id)
        return {
            "status": "failed",
            "message": "Project not found.",
            "source_counts": {},
            "total_chunks": 0,
        }

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
    papers_enabled = bool(project.get("papers_enabled"))
    patents_enabled = bool(project.get("patents_enabled"))
    perigon_enabled = bool(project.get("perigon_enabled"))
    tavily_enabled = bool(project.get("tavily_enabled"))
    exa_enabled = bool(project.get("exa_enabled"))

    if not social_filter:
        logger.warning(
            "ingest_project.no_social_filter",
            project_id=project_id,
            project_title=project_title[:80],
        )
        instagram_enabled = False
        tiktok_enabled = False

    logger.info(
        "ingest_project.start",
        project_id=project_id,
        social_filter=social_filter[:60],
        tiktok_enabled=tiktok_enabled,
        instagram_enabled=instagram_enabled,
        papers_enabled=papers_enabled,
        patents_enabled=patents_enabled,
        perigon_enabled=perigon_enabled,
        tavily_enabled=tavily_enabled,
        exa_enabled=exa_enabled,
    )

    source_types = [
        "social_tiktok",
        "social_instagram",
        "paper",
        "paper",
        "paper",
        "paper",
        "patent",
        "patent",
        "patent",
        "news",
        "search",
        "search",
    ]
    source_enabled_flags = [
        tiktok_enabled,
        instagram_enabled,
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

    gather_results = await asyncio.gather(
        _run_source_with_timeout(
            "social_tiktok",
            _ingest_tiktok(project_id=project_id, user_id=user_id, social_filter=social_filter),
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
                oldest_timestamp=oldest_timestamp,
                redis=redis,
            ),
        )
        if instagram_enabled
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
        )
        if patents_enabled
        else _noop(),
        _run_source_with_timeout(
            "news_perigon",
            _ingest_news(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                redis=redis,
            ),
        )
        if perigon_enabled
        else _noop(),
        _run_source_with_timeout(
            "search_tavily",
            _ingest_web_tavily(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                redis=redis,
            ),
        )
        if tavily_enabled
        else _noop(),
        _run_source_with_timeout(
            "search_exa",
            _ingest_web_exa(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                redis=redis,
            ),
        )
        if exa_enabled
        else _noop(),
    )

    intent_text = json.dumps(intent)
    documents: list[RawDocument] = []
    any_source_completed = False

    for result, source, source_enabled in zip(
        gather_results, source_types, source_enabled_flags, strict=False
    ):
        if not source_enabled:
            continue
        if isinstance(result, BaseException):
            logger.warning(
                "ingest_tool.failed", project_id=project_id, source=source, error=str(result)
            )
            continue
        any_source_completed = True
        if not result:
            continue

        if oldest_timestamp is not None:
            result = [doc for doc in result if _is_document_new_enough(doc, oldest_timestamp)]
            if not result:
                continue

        filter_items: list[dict] = []
        for index, doc in enumerate(result):
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
                }
            )

        filtered_items = await _filter_relevance(
            filter_items,
            intent_text,
            source,
            redis,
            must_match_terms=intent.get("must_match_terms") or [],
        )
        kept_indexes = {
            int(item.get("_idx", -1))
            for item in filtered_items
            if isinstance(item, dict) and isinstance(item.get("_idx"), int)
        }
        documents.extend(doc for index, doc in enumerate(result) if index in kept_indexes)

    if not documents:
        logger.warning("ingest_project.no_documents", project_id=project_id)
        if any_source_completed:
            async with pool.acquire() as conn:
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM knowledge_chunks WHERE project_id = $1::uuid",
                    project_id,
                )
            await project_repo.update_project_kb_stats(
                pool,
                project_id,
                int(total or 0),
                datetime.now(UTC),
            )
            return {
                "status": "empty" if int(total or 0) <= 0 else "ready",
                "message": "Ingestion completed but no new documents passed filters.",
                "source_counts": {},
                "total_chunks": int(total or 0),
            }
        return {
            "status": "failed",
            "message": "No source completed successfully.",
            "source_counts": {},
            "total_chunks": 0,
        }

    inserted = await ingest_documents(documents)

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE project_id = $1::uuid",
            project_id,
        )

    await project_repo.update_project_kb_stats(pool, project_id, int(total or 0), datetime.now(UTC))
    await _invalidate_project_caches(project_id)

    logger.info(
        "ingest_project.done",
        project_id=project_id,
        social_instagram_docs=sum(1 for d in documents if d.source == "social_instagram"),
        social_tiktok_docs=sum(1 for d in documents if d.source == "social_tiktok"),
        paper_docs=sum(1 for d in documents if d.source == "paper"),
        patent_docs=sum(1 for d in documents if d.source == "patent"),
        news_docs=sum(1 for d in documents if d.source == "news"),
        search_docs=sum(1 for d in documents if d.source == "search"),
        total_docs=len(documents),
        chunks_inserted=inserted,
        total_chunks=total,
    )
    source_counts = {
        "social_instagram": sum(1 for d in documents if d.source == "social_instagram"),
        "social_tiktok": sum(1 for d in documents if d.source == "social_tiktok"),
        "paper": sum(1 for d in documents if d.source == "paper"),
        "patent": sum(1 for d in documents if d.source == "patent"),
        "news": sum(1 for d in documents if d.source == "news"),
        "search": sum(1 for d in documents if d.source == "search"),
    }
    return {
        "status": "ready",
        "message": "Ingestion completed.",
        "source_counts": source_counts,
        "total_chunks": int(total or 0),
    }


async def _ingest_instagram(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    oldest_timestamp: int | None,
    redis,
) -> list[RawDocument]:
    """Ingest Instagram reels via hashtag-first discovery, fail-open on errors."""
    _ = redis
    handle_candidates = _instagram_handle_candidates(social_filter, intent)
    hashtags = _extract_hashtags(social_filter)
    if not hashtags:
        for keyword in intent.get("keywords") or []:
            cleaned = _clean_instagram_keyword_query(str(keyword))
            if cleaned:
                hashtags.append(cleaned)
            if len(hashtags) >= 5:
                break
    if not hashtags:
        fallback_reels = await _fetch_instagram_reels_from_candidates(
            project_id=project_id,
            candidate_queries=handle_candidates,
            oldest_timestamp=oldest_timestamp,
        )
        if not fallback_reels:
            logger.warning("ingest_instagram.no_hashtags", project_id=project_id)
            return []
        all_posts: list[dict] = fallback_reels
    else:
        all_posts = []

    logger.info("ingest_instagram.start", project_id=project_id, hashtag_count=len(hashtags))

    now_utc = datetime.now(UTC)
    cutoff_30d = now_utc - timedelta(days=30)
    oldest_cutoff = (
        datetime.fromtimestamp(oldest_timestamp, tz=UTC)
        if isinstance(oldest_timestamp, int | float) and oldest_timestamp > 0
        else None
    )
    recency_cutoff = max(cutoff_30d, oldest_cutoff) if oldest_cutoff else cutoff_30d

    if hashtags:
        max_pages = max(1, settings.INGEST_INSTAGRAM_HASHTAG_PAGES)
        for hashtag in hashtags[:5]:
            cursor: str | None = None
            for _ in range(max_pages):
                try:
                    posts, next_cursor = await instagram_hashtag_posts(
                        hashtag=hashtag.replace(" ", ""),
                        cursor=cursor,
                        get_author_info=True,
                    )
                except Exception:
                    logger.exception(
                        "ingest_instagram.hashtag_failed",
                        project_id=project_id,
                        hashtag=hashtag,
                    )
                    break

                for post in posts:
                    ts = post.get("timestamp")
                    if isinstance(ts, int | float):
                        post_dt = datetime.fromtimestamp(float(ts), tz=UTC)
                        if post_dt < recency_cutoff:
                            continue
                    all_posts.append(post)

                cursor = next_cursor
                if not cursor:
                    break

    if not all_posts:
        fallback_reels = await _fetch_instagram_reels_from_candidates(
            project_id=project_id,
            candidate_queries=handle_candidates,
            oldest_timestamp=oldest_timestamp,
        )
        if not fallback_reels:
            logger.warning("ingest_instagram.no_posts", project_id=project_id)
            return []
        all_posts.extend(fallback_reels)

    author_scores: dict[str, int] = {}
    for post in all_posts:
        username = str(post.get("username") or "").strip().lower()
        if not username:
            continue
        engagement = _as_int(post.get("like_count")) + _as_int(post.get("view_count"))
        author_scores[username] = author_scores.get(username, 0) + engagement

    top_authors = sorted(
        author_scores, key=lambda author: author_scores.get(author, 0), reverse=True
    )[:10]
    author_id_map: dict[str, int] = {}
    for username in top_authors:
        try:
            search_results = await instagram_search(username)
        except Exception:
            logger.exception(
                "ingest_instagram.author_lookup_failed",
                project_id=project_id,
                username=username,
            )
            continue
        exact_match = next(
            (
                item
                for item in search_results
                if str(item.get("username") or "").strip().lower() == username
            ),
            None,
        )
        raw_uid = (exact_match or {}).get("pk") or (exact_match or {}).get("id")
        if not raw_uid:
            continue
        try:
            author_id_map[username] = int(str(raw_uid))
        except (TypeError, ValueError):
            continue

    all_reels: list[dict] = list(all_posts)
    for username, uid in author_id_map.items():
        try:
            reels = await instagram_user_reels(uid, depth=1, oldest_timestamp=oldest_timestamp)
            for reel in reels:
                if not reel.get("username"):
                    reel["username"] = username
            all_reels.extend(reels or [])
        except Exception:
            logger.exception(
                "ingest_instagram.reels_failed",
                project_id=project_id,
                username=username,
                uid=uid,
            )

    relevance_terms = _build_relevance_terms(intent, social_filter)
    brand_terms = _build_brand_terms(intent)

    scored: list[tuple[float, dict]] = []
    for reel in all_reels:
        caption = str(reel.get("caption") or "").strip()
        if not caption:
            continue

        likes = _as_int(reel.get("like_count"))
        views = _as_int(reel.get("view_count") or reel.get("play_count"))
        relevance = float(
            _match_count(caption, relevance_terms) + _match_count(caption, brand_terms)
        )
        engagement_score = min(1.0, math.log1p(max(0, likes + views)) / 20.0)

        days_old = 15.0
        ts = reel.get("timestamp")
        if isinstance(ts, int | float):
            try:
                reel_dt = datetime.fromtimestamp(float(ts), tz=UTC)
                days_old = max(0.0, (now_utc - reel_dt).total_seconds() / 86400.0)
            except Exception:
                days_old = 15.0
        recency_score = math.exp(-days_old / REEL_RECENCY_DAYS)
        total_score = (relevance * 0.60) + (engagement_score * 0.25) + (recency_score * 0.15)
        scored.append((total_score, reel))

    scored.sort(key=lambda item: item[0], reverse=True)

    docs: list[RawDocument] = []
    seen_ids: set[str] = set()
    limit = max(1, settings.INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT)
    for _, reel in scored:
        source_id = str(reel.get("shortcode") or "").strip()
        if not source_id or source_id in seen_ids:
            continue
        caption = str(reel.get("caption") or "").strip()
        if not caption:
            continue

        username = str(reel.get("username") or "").strip() or "instagram"
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_instagram",
                source_id=source_id,
                title=f"@{username}",
                content=caption[:2000],
                metadata={
                    "platform": "instagram",
                    "author": username,
                    "likes": _as_int(reel.get("like_count")),
                    "views": _as_int(reel.get("view_count") or reel.get("play_count")),
                    "timestamp": reel.get("timestamp"),
                    "cover_url": reel.get("cover_url") or reel.get("display_url") or "",
                    "video_url": reel.get("video_url") or "",
                    "url": f"https://www.instagram.com/reel/{source_id}/",
                    "tool": "instagram_hashtag_first",
                },
            )
        )
        seen_ids.add(source_id)
        if len(docs) >= limit:
            break

    logger.info(
        "ingest_instagram.success",
        project_id=project_id,
        hashtags=len(hashtags),
        hashtag_posts=len(all_posts),
        candidate_reels=len(all_reels),
        docs=len(docs),
    )
    return docs


async def _ingest_tiktok(
    *,
    project_id: str,
    user_id: str,
    social_filter: str,
) -> list[RawDocument]:
    """
    Fetch recent TikTok posts for the handle derived from social_filter.

    social_filter is reused as the TikTok handle (same structured_intent field).
    Videos with an empty description are skipped — they carry no KB value.

    Returns a list of RawDocument with source="social_tiktok".
    Returns [] (and logs a warning) if the fetch yields no posts.
    Never raises — all exceptions are caught and logged.
    """
    handle = social_filter
    posts = await fetch_tiktok_posts(handle)
    logger.info(
        "ingest_project.tiktok.posts_fetched",
        project_id=project_id,
        handle=handle,
        post_count=len(posts),
    )

    docs: list[RawDocument] = []
    for video in posts:
        description = (video.get("description") or "").strip()
        if not description:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_tiktok",
                source_id=video.get("video_id", ""),
                title=f"@{video.get('author_username', '')}",
                content=description,
                metadata={
                    "platform": "tiktok",
                    "author_username": video.get("author_username", ""),
                    "likes": video.get("likes", 0),
                    "views": video.get("views", 0),
                    "cover_url": video.get("cover_url", ""),
                    "video_url": video.get("video_url", ""),
                    "timestamp": video.get("create_time") or video.get("created_at"),
                },
            )
        )

    return docs


async def _ingest_papers_openalex(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
) -> list[RawDocument]:
    """
    Fetch research papers from OpenAlex using the best available query string.

    Query priority:
    1. structured_intent.search_filters.papers — natural language academic query
    2. Fallback: social_filter with hashtag symbols stripped

    The old code read intent.get("description") / intent.get("query") — neither
    field exists in structured_intent — so it always fell through to the raw
    social_filter hashtag string (e.g. "#PG #beautytrends"), which OpenAlex
    cannot match. This fix reads the correct field.

    Papers with no abstract are skipped — an empty abstract produces useless KB chunks.

    Returns a list of RawDocument with source="paper".
    Returns [] (and logs a warning) if the fetch yields no papers.
    Never raises — all exceptions are caught and logged.
    """
    query = _query_for_papers(
        intent,
        source="openalex",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_openalex.no_query", project_id=project_id)
        return []

    papers = await fetch_papers(query)
    logger.info(
        "ingest_project.openalex.papers_fetched",
        project_id=project_id,
        query_preview=query[:80],
        paper_count=len(papers),
    )

    docs: list[RawDocument] = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=paper.get("paper_id", ""),
                title=paper.get("title", ""),
                content=abstract,
                metadata={
                    "doi": paper.get("doi", ""),
                    "publication_year": paper.get("publication_year", 0),
                    "tool": "openalex",
                },
            )
        )

    return docs


async def _ingest_papers_semantic_scholar(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch papers from Semantic Scholar and map to RawDocument."""
    _ = redis
    query = _query_for_papers(
        intent,
        source="semantic_scholar",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_ss.no_query", project_id=project_id)
        return []

    try:
        papers = await search_semantic_scholar(
            query,
            must_match_terms=intent.get("must_match_terms") or [],
            domain_terms=intent.get("domain_terms") or [],
            query_specificity=intent.get("query_specificity"),
        )
    except Exception:
        logger.exception("ingest_ss.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for paper in papers:
        content = str(paper.get("content") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=str(paper.get("paper_id") or paper.get("url") or ""),
                title=str(paper.get("title") or ""),
                content=content,
                metadata={
                    "url": paper.get("url") or "",
                    "authors": paper.get("authors") or [],
                    "year": paper.get("year"),
                    "citation_count": paper.get("citation_count", 0),
                    "tool": "semantic_scholar",
                },
            )
        )

    logger.info("ingest_ss.success", project_id=project_id, count=len(docs))
    return docs


async def _ingest_papers_pubmed(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch papers from PubMed and map to RawDocument."""
    _ = redis
    query = _query_for_papers(
        intent,
        source="pubmed",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_pubmed.no_query", project_id=project_id)
        return []

    try:
        papers = await search_pubmed(
            query,
            must_match_terms=intent.get("must_match_terms") or [],
            domain_terms=intent.get("domain_terms") or [],
            query_specificity=intent.get("query_specificity"),
        )
    except Exception:
        logger.exception("ingest_pubmed.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for paper in papers:
        content = str(paper.get("content") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=str(paper.get("pmid") or paper.get("url") or ""),
                title=str(paper.get("title") or ""),
                content=content,
                metadata={
                    "pmid": paper.get("pmid") or "",
                    "journal": paper.get("journal") or "",
                    "published": paper.get("published") or "",
                    "authors": paper.get("authors") or [],
                    "url": paper.get("url") or "",
                    "tool": "pubmed",
                },
            )
        )

    logger.info("ingest_pubmed.success", project_id=project_id, count=len(docs))
    return docs


async def _ingest_papers_arxiv(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch papers from arXiv and map to RawDocument."""
    _ = redis
    query = _query_for_papers(
        intent,
        source="arxiv",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_arxiv.no_query", project_id=project_id)
        return []

    try:
        papers = await search_arxiv(
            query,
            must_match_terms=intent.get("must_match_terms") or [],
            domain_terms=intent.get("domain_terms") or [],
            query_specificity=intent.get("query_specificity"),
        )
    except Exception:
        logger.exception("ingest_arxiv.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for paper in papers:
        content = str(paper.get("content") or paper.get("summary") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=str(paper.get("arxiv_id") or paper.get("url") or ""),
                title=str(paper.get("title") or ""),
                content=content,
                metadata={
                    "arxiv_id": paper.get("arxiv_id") or "",
                    "authors": paper.get("authors") or [],
                    "published": paper.get("published") or "",
                    "categories": paper.get("categories") or [],
                    "url": paper.get("url") or "",
                    "tool": "arxiv",
                },
            )
        )

    logger.info("ingest_arxiv.success", project_id=project_id, count=len(docs))
    return docs


async def _ingest_patents_patentsview(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch patents from PatentsView and map to RawDocument."""
    _ = redis
    query = _query_for_patents(
        intent,
        source="patentsview",
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_patentsview.no_query", project_id=project_id)
        return []

    try:
        patents = await search_patentsview(
            query,
            must_match_terms=intent.get("must_match_terms") or [],
            domain_terms=intent.get("domain_terms") or [],
            query_specificity=intent.get("query_specificity"),
        )
    except Exception:
        logger.exception("ingest_patentsview.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for patent in patents:
        abstract = str(patent.get("content") or "").strip()
        if not abstract:
            continue
        claims_snippet = abstract[:400]
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="patent",
                source_id=str(patent.get("patent_number") or patent.get("url") or ""),
                title=str(patent.get("title") or ""),
                content=abstract,
                metadata={
                    "patent_number": patent.get("patent_number") or "",
                    "date": patent.get("date") or "",
                    "inventors": patent.get("inventors") or [],
                    "url": patent.get("url") or "",
                    "claims_snippet": claims_snippet,
                    "tool": "patentsview",
                },
            )
        )

    logger.info("ingest_patentsview.success", project_id=project_id, count=len(docs))
    return docs


async def _ingest_patents_lens(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch patents from Lens and map to RawDocument."""
    _ = redis
    query = _query_for_patents(
        intent,
        source="lens",
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_lens.no_query", project_id=project_id)
        return []

    try:
        patents = await search_lens(query)
    except Exception:
        logger.exception("ingest_lens.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for patent in patents:
        abstract = str(patent.get("content") or "").strip()
        if not abstract:
            continue
        claims_snippet = abstract[:400]
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="patent",
                source_id=str(patent.get("lens_id") or patent.get("patent_number") or ""),
                title=str(patent.get("title") or ""),
                content=abstract,
                metadata={
                    "lens_id": patent.get("lens_id") or "",
                    "patent_number": patent.get("patent_number") or "",
                    "jurisdiction": patent.get("jurisdiction") or "",
                    "kind": patent.get("kind") or "",
                    "date": patent.get("date") or "",
                    "inventors": patent.get("inventors") or [],
                    "url": patent.get("url") or "",
                    "claims_snippet": claims_snippet,
                    "tool": "lens",
                },
            )
        )

    logger.info("ingest_lens.success", project_id=project_id, count=len(docs))
    return docs


def _query_for_news_or_web(intent: dict, *, source: str = "perigon") -> str:
    """Select the best available query string for news/web tools."""
    search_filters = intent.get("search_filters") or {}
    entities = intent.get("entities") or []
    keywords = intent.get("keywords") or []
    seed_query = (
        search_filters.get("news")
        or search_filters.get("papers")
        or (entities[0] if entities else "")
        or " ".join(str(k) for k in keywords[:4])
    )
    policy = build_query_policy(str(seed_query or "").strip(), intent)
    return build_source_query(policy, source)


def _project_context_query(project_title: str, project_description: str) -> str:
    """Create a short fallback query from project title/description."""
    title = str(project_title or "").strip()
    description = str(project_description or "").strip()
    if not title and not description:
        return ""
    text = f"{title} {description}".strip()
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return " ".join(tokens[:12]).strip()


def _query_for_social(
    intent: dict,
    *,
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Select a robust social search seed even when intent filters are sparse."""
    search_filters = intent.get("search_filters") or {}
    keywords = intent.get("keywords") or []
    entities = intent.get("entities") or []

    direct = str(search_filters.get("social") or search_filters.get("instagram") or "").strip()
    if direct:
        return direct

    terms = [str(value).strip() for value in keywords[:4] if str(value).strip()]
    if not terms:
        terms = [str(value).strip() for value in entities[:3] if str(value).strip()]
    if terms:
        return " ".join(terms)

    return _project_context_query(project_title, project_description)


def _query_for_papers(
    intent: dict,
    *,
    source: str = "openalex",
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Select the best available paper query string with safe fallbacks."""
    search_filters = intent.get("search_filters") or {}
    entities = intent.get("entities") or []
    keywords = intent.get("keywords") or []

    direct = str(search_filters.get("papers") or "").strip()
    if direct:
        policy = build_query_policy(direct, intent)
        return build_source_query(policy, source)

    news_fallback = str(search_filters.get("news") or "").strip()
    if news_fallback:
        policy = build_query_policy(news_fallback, intent)
        return build_source_query(policy, source)

    terms = [str(value).strip() for value in entities[:2] if str(value).strip()]
    terms.extend(str(value).strip() for value in keywords[:6] if str(value).strip())
    if terms:
        policy = build_query_policy(" ".join(terms), intent)
        return build_source_query(policy, source)

    social_fallback = str(social_filter or "").replace("#", " ").strip()
    if social_fallback:
        policy = build_query_policy(social_fallback, intent)
        return build_source_query(policy, source)

    fallback = _project_context_query(project_title, project_description)
    policy = build_query_policy(fallback, intent)
    return build_source_query(policy, source)


def _query_for_patents(
    intent: dict,
    *,
    source: str = "patentsview",
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Select the best available query string for patent tools."""
    search_filters = intent.get("search_filters") or {}
    entities = intent.get("entities") or []
    keywords = intent.get("keywords") or []

    direct = str(search_filters.get("patents") or search_filters.get("papers") or "").strip()
    if direct:
        policy = build_query_policy(direct, intent)
        return build_source_query(policy, source)

    terms: list[str] = []
    if entities:
        terms.append(str(entities[0]))
    terms.extend(str(keyword) for keyword in keywords[:4] if str(keyword).strip())
    if terms:
        policy = build_query_policy(
            " ".join(term.strip() for term in terms if term.strip()),
            intent,
        )
        return build_source_query(policy, source)

    fallback = _project_context_query(project_title, project_description)
    policy = build_query_policy(fallback, intent)
    return build_source_query(policy, source)


def _looks_like_patent_url(url: str) -> bool:
    """Return True when a URL likely points to a patent record/document."""
    lowered = (url or "").strip().lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "patents.google.com/patent/",
            "patentscope.wipo.int",
            "uspto.gov/patents",
            "worldwide.espacenet.com/patent",
            "lens.org/lens/patent",
            "/patent/",
        )
    )


async def _ingest_patents_web_fallback(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
) -> list[RawDocument]:
    """Fallback patent ingest via web search when patent APIs return no data."""
    query = _query_for_patents(
        intent,
        source="patentsview",
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        return []

    candidates: list[dict] = []
    try:
        candidates.extend(await search_exa(f"{query} patents"))
    except Exception:
        logger.exception("ingest_patent_fallback.exa_failed", project_id=project_id)
    try:
        candidates.extend(await search_tavily(f"{query} patents"))
    except Exception:
        logger.exception("ingest_patent_fallback.tavily_failed", project_id=project_id)

    seen_urls: set[str] = set()
    docs: list[RawDocument] = []
    for item in candidates:
        url = str(item.get("url") or "").strip()
        if not url or url in seen_urls or not _looks_like_patent_url(url):
            continue
        seen_urls.add(url)

        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or item.get("snippet") or "").strip()
        if not content:
            continue

        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="patent",
                source_id=url,
                title=title or "Patent result",
                content=content,
                metadata={
                    "url": url,
                    "tool": "patent_web_fallback",
                    "score": item.get("score", 0.0),
                },
            )
        )
        if len(docs) >= 12:
            break

    logger.info(
        "ingest_patent_fallback.done",
        project_id=project_id,
        candidate_count=len(candidates),
        doc_count=len(docs),
    )
    return docs


async def _ingest_news(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    redis,
) -> list[RawDocument]:
    """Fetch news from Perigon and map to RawDocument."""
    _ = redis
    query = _query_for_news_or_web(intent, source="perigon")
    if not query:
        logger.warning("ingest_news.no_query", project_id=project_id)
        return []

    try:
        articles = await search_perigon(
            query,
            must_match_terms=intent.get("must_match_terms") or [],
            domain_terms=intent.get("domain_terms") or [],
            query_specificity=intent.get("query_specificity"),
        )
    except Exception:
        logger.exception("ingest_news.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for article in articles:
        content = str(article.get("content") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="news",
                source_id=str(article.get("url") or article.get("title") or ""),
                title=str(article.get("title") or ""),
                content=content,
                metadata={
                    "url": article.get("url") or "",
                    "source_name": article.get("source_name") or "",
                    "published_at": article.get("published_date") or "",
                    "tool": "perigon",
                },
            )
        )

    logger.info("ingest_news.success", project_id=project_id, count=len(docs))
    return docs


async def _ingest_web_tavily(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    redis,
) -> list[RawDocument]:
    """Fetch web results from Tavily and map to RawDocument."""
    _ = redis
    query = _query_for_news_or_web(intent, source="tavily")
    if not query:
        logger.warning("ingest_tavily.no_query", project_id=project_id)
        return []

    try:
        results = await search_tavily(query)
    except Exception:
        logger.exception("ingest_tavily.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for result in results:
        content = str(result.get("content") or result.get("snippet") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="search",
                source_id=str(result.get("url") or result.get("title") or ""),
                title=str(result.get("title") or ""),
                content=content,
                metadata={
                    "url": result.get("url") or "",
                    "score": result.get("score", 0.0),
                    "published_at": result.get("published_date") or "",
                    "tool": "tavily",
                },
            )
        )

    logger.info("ingest_tavily.success", project_id=project_id, count=len(docs))
    return docs


async def _ingest_web_exa(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    redis,
) -> list[RawDocument]:
    """Fetch web results from Exa and map to RawDocument."""
    _ = redis
    query = _query_for_news_or_web(intent, source="exa")
    if not query:
        logger.warning("ingest_exa.no_query", project_id=project_id)
        return []

    try:
        results = await search_exa(query)
    except Exception:
        logger.exception("ingest_exa.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for result in results:
        content = str(result.get("content") or result.get("snippet") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="search",
                source_id=str(result.get("url") or result.get("title") or ""),
                title=str(result.get("title") or ""),
                content=content,
                metadata={
                    "url": result.get("url") or "",
                    "score": result.get("score", 0.0),
                    "published_at": result.get("published_date") or "",
                    "authors": result.get("authors") or [],
                    "tool": "exa",
                },
            )
        )

    logger.info("ingest_exa.success", project_id=project_id, count=len(docs))
    return docs


async def _ingest_openalex(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
) -> list[RawDocument]:
    """Backward-compatible alias for existing callers."""
    return await _ingest_papers_openalex(
        project_id=project_id,
        user_id=user_id,
        intent=intent,
        social_filter=social_filter,
    )
