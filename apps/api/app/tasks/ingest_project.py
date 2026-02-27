"""ingest_project ARQ task — initial KB population from enabled sources after project creation.

Called immediately after project creation via ARQ enqueue.
Also re-used by refresh_project for subsequent refreshes.

Supports independently toggleable ingestion sources per project:
  - Social: Instagram, TikTok, YouTube, Reddit, X
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

from app.core.cache_version import bump_project_cache_version
from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.db import get_db_pool
from app.core.metrics import fulltext_resolve_total
from app.core.openai_client import get_openai_client
from app.core.query_policy import lexical_entity_coverage
from app.core.redis import get_redis
from app.kb.embedder import embed_texts
from app.kb.ingester import RawDocument, ingest_documents
from app.repositories import project as project_repo
from app.repositories import source_asset as source_asset_repo
from app.services.fulltext_resolver import resolve_fulltext_url
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
from app.tools.social_reddit import search_reddit_posts
from app.tools.social_tiktok import fetch_tiktok_posts
from app.tools.social_x import search_x_posts
from app.tools.social_youtube import search_youtube_videos

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
RELEVANCE_MIN_STAGE2_KEEP_SOCIAL = 6
SOCIAL_SOURCES: frozenset[str] = frozenset(
    {
        "social_tiktok",
        "social_instagram",
        "social_youtube",
        "social_reddit",
        "social_x",
    }
)

async def _noop() -> list[RawDocument]:
    """Return an empty source result for disabled gather slots."""
    return []


async def _run_source_with_timeout(
    source: str,
    coro,
    *,
    timeout_seconds: float | None = None,
) -> list[RawDocument]:
    """Run one source ingest with timeout and fail-open behavior."""
    timeout = float(timeout_seconds or settings.INGEST_SOURCE_TIMEOUT)
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        logger.warning(
            "ingest_tool.timeout",
            source=source,
            timeout_seconds=timeout,
        )
        return []
    except Exception as exc:
        logger.warning("ingest_tool.failed", source=source, error=str(exc))
        return []


def _dedupe_social_docs(docs: list[RawDocument]) -> list[RawDocument]:
    """Deduplicate social documents by (source, source_id) while preserving order."""
    out: list[RawDocument] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        source = str(doc.source or "").strip()
        source_id = str(doc.source_id or "").strip()
        if not source_id:
            continue
        key = (source, source_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _social_web_domain_tokens(source: str) -> tuple[str, ...]:
    """Return domain tokens used to validate web-fallback URLs per social source."""
    return {
        "social_youtube": ("youtube.com/", "youtu.be/"),
        "social_reddit": ("reddit.com/r/", "reddit.com/comments/"),
        "social_x": ("x.com/", "twitter.com/"),
        "social_instagram": ("instagram.com/reel/", "instagram.com/p/"),
        "social_tiktok": ("tiktok.com/",),
    }.get(source, ())


def _social_web_query(source: str, base_query: str) -> str:
    """Build a site-constrained web query for social fallback retrieval."""
    q = str(base_query or "").strip()
    if source == "social_youtube":
        return f"{q} site:youtube.com"
    if source == "social_reddit":
        return f"{q} site:reddit.com/r"
    if source == "social_x":
        return f"{q} (site:x.com OR site:twitter.com)"
    if source == "social_instagram":
        return f"{q} site:instagram.com/reel"
    if source == "social_tiktok":
        return f"{q} site:tiktok.com"
    return q


async def _social_web_fallback_docs(
    *,
    source: str,
    project_id: str,
    user_id: str,
    query: str,
    keep_limit: int,
) -> list[RawDocument]:
    """Fallback social ingestion from web search when native social APIs return no rows."""
    constrained_query = _social_web_query(source, query)
    if not constrained_query:
        return []

    candidates: list[dict] = []
    try:
        candidates.extend(await search_exa(constrained_query))
    except Exception:
        logger.exception("social_web_fallback.exa_failed", source=source, project_id=project_id)
    try:
        candidates.extend(await search_tavily(constrained_query))
    except Exception:
        logger.exception("social_web_fallback.tavily_failed", source=source, project_id=project_id)

    domain_tokens = _social_web_domain_tokens(source)
    docs: list[RawDocument] = []
    seen_urls: set[str] = set()
    for row in candidates:
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        lowered_url = url.lower()
        if domain_tokens and not any(token in lowered_url for token in domain_tokens):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = str(row.get("title") or "").strip() or "Social result"
        content = str(row.get("content") or row.get("snippet") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source=source,
                source_id=url,
                title=title[:240],
                content=content[:2000],
                metadata={
                    "platform": source.removeprefix("social_"),
                    "url": url,
                    "published_at": row.get("published_date") or "",
                    "tool": f"{row.get('tool') or 'web'}_social_fallback",
                },
            )
        )
        if len(docs) >= max(1, keep_limit):
            break

    logger.info(
        "social_web_fallback.done",
        source=source,
        project_id=project_id,
        query_preview=constrained_query[:100],
        candidate_count=len(candidates),
        doc_count=len(docs),
    )
    return docs


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
        "source_diagnostics": source_diagnostics or {},
        "fulltext_enqueued": int(fulltext_enqueued),
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


def _clean_term_values(values: list[str] | None) -> list[str]:
    """Normalize term strings while preserving order and uniqueness."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        cleaned = str(value or "").strip().lower()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


async def _filter_social_strict(
    items: list[dict],
    intent_text: str,
    *,
    source: str,
    must_match_terms: list[str] | None = None,
    social_match_terms: list[str] | None = None,
) -> list[dict]:
    """
    Strict deterministic relevance gate for social content.

    Rules:
    - For specific queries (must_match_terms present), item text must pass lexical entity coverage.
    - For all social queries, item text must pass minimum embedding similarity.
    - If embedding fails, keep only lexical-pass items (never broad fail-open for social).
    """
    if not items:
        return items

    must_terms = _clean_term_values(must_match_terms)
    if not must_terms:
        must_terms = _clean_term_values(social_match_terms)
    required_term_hits = _required_must_match_count(must_terms)
    lexical_pass: list[dict] = []
    min_coverage = 0.0
    if must_terms:
        min_coverage = required_term_hits / max(1, len(must_terms))
    for item in items:
        text = _relevance_item_text(item)
        if must_terms:
            coverage = lexical_entity_coverage(text, must_terms)
            if coverage < min_coverage:
                continue
        lexical_pass.append(item)

    if not lexical_pass:
        logger.info(
            "relevance_filter.social_lexical",
            source=source,
            kept=0,
            total=len(items),
            must_term_count=len(must_terms),
        )
        return []

    try:
        intent_embeddings = await embed_texts([intent_text])
        if not intent_embeddings:
            return lexical_pass
        intent_arr = np.array(intent_embeddings[0], dtype=float)
        intent_norm = float(np.linalg.norm(intent_arr))
        if intent_norm <= 0:
            return lexical_pass

        texts = [_relevance_item_text(item) for item in lexical_pass]
        item_embeddings = await embed_texts(texts)
        if len(item_embeddings) != len(lexical_pass):
            logger.warning(
                "relevance_filter.social_embedding_mismatch",
                source=source,
                expected=len(lexical_pass),
                actual=len(item_embeddings),
            )
            return lexical_pass

        kept: list[dict] = []
        for item, embedding in zip(lexical_pass, item_embeddings, strict=False):
            item_arr = np.array(embedding, dtype=float)
            denom = intent_norm * float(np.linalg.norm(item_arr))
            if denom <= 0:
                continue
            score = float(np.dot(intent_arr, item_arr) / denom)
            if score < settings.SOCIAL_RELEVANCE_MIN_SIMILARITY:
                continue
            item["_social_similarity"] = round(score, 6)
            kept.append(item)

        logger.info(
            "relevance_filter.social_strict",
            source=source,
            kept=len(kept),
            lexical_kept=len(lexical_pass),
            total=len(items),
            min_similarity=settings.SOCIAL_RELEVANCE_MIN_SIMILARITY,
        )
        if not kept:
            fallback_keep = min(RELEVANCE_MIN_STAGE2_KEEP_SOCIAL, len(lexical_pass))
            logger.info(
                "relevance_filter.social_similarity_fallback",
                source=source,
                lexical_kept=len(lexical_pass),
                fallback_keep=fallback_keep,
                min_similarity=settings.SOCIAL_RELEVANCE_MIN_SIMILARITY,
            )
            return lexical_pass[:fallback_keep]
        try:
            stage2_social = await _filter_stage2_llm_social(
                kept,
                intent_text=intent_text,
                source=source,
                must_match_terms=must_terms,
            )
            min_keep = min(RELEVANCE_MIN_STAGE2_KEEP_SOCIAL, len(kept))
            if len(stage2_social) < min_keep:
                logger.info(
                    "relevance_filter.stage2_social_min_fallback",
                    source=source,
                    llm_kept=len(stage2_social),
                    min_keep=min_keep,
                )
                ranked_kept = sorted(
                    kept,
                    key=lambda item: float(item.get("_social_similarity", 0.0)),
                    reverse=True,
                )
                return ranked_kept[:min_keep]
            return stage2_social
        except Exception:
            logger.exception("relevance_filter.stage2_social_failed", source=source)
            return kept
    except Exception:
        logger.exception(
            "relevance_filter.social_embedding_failed",
            source=source,
            lexical_kept=len(lexical_pass),
        )
        return lexical_pass


async def _filter_relevance(
    items: list[dict],
    intent_text: str,
    source: str,
    redis,
    must_match_terms: list[str] | None = None,
    social_match_terms: list[str] | None = None,
) -> list[dict]:
    """Apply two-stage relevance filtering with fail-open behavior."""
    if not items:
        return items

    if source in SOCIAL_SOURCES:
        return await _filter_social_strict(
            items,
            intent_text,
            source=source,
            must_match_terms=must_match_terms,
            social_match_terms=social_match_terms,
        )

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

    client = get_openai_client()
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
                    lexical_match = int(_match_count(text, set(_clean_term_values(must_terms))))
                    if lexical_match < _required_must_match_count(must_terms):
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


async def _filter_stage2_llm_social(
    items: list[dict],
    *,
    intent_text: str,
    source: str,
    must_match_terms: list[str] | None = None,
) -> list[dict]:
    """Use batched GPT-4o-mini relevance judging for social items."""
    if not items:
        return items
    if not settings.INGEST_SOCIAL_LLM_FILTER_ENABLED:
        return items
    if not settings.OPENAI_API_KEY:
        return items

    ranked_items = sorted(
        items,
        key=lambda item: float(item.get("_social_similarity", 0.0)),
        reverse=True,
    )
    capped = ranked_items[: max(1, settings.INGEST_SOCIAL_LLM_MAX_CANDIDATES)]
    client = get_openai_client()
    kept_items: list[dict] = []

    for batch_start in range(0, len(capped), RELEVANCE_STAGE2_BATCH_SIZE):
        batch = capped[batch_start : batch_start + RELEVANCE_STAGE2_BATCH_SIZE]
        numbered: list[str] = []
        for index, item in enumerate(batch):
            title = str(item.get("title") or "Unknown title")
            body_text = str(item.get("content") or item.get("abstract") or item.get("claims") or "")
            numbered.append(f"{index}. Title: {title}\nContent: {body_text[:500]}")

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
            f"Source type: {source}\n"
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
                    "relevance_filter.stage2_social_invalid_shape",
                    source=source,
                    batch_size=len(batch),
                )
                if settings.INGEST_SOCIAL_LLM_FAIL_OPEN:
                    kept_items.extend(batch)
                continue

            relevant_ids: set[int] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    item_id = int(str(entry.get("id")))
                except (TypeError, ValueError):
                    continue
                if int(str(entry.get("relevant", 0))) == 1:
                    relevant_ids.add(item_id)

            kept_items.extend(item for index, item in enumerate(batch) if index in relevant_ids)
        except Exception:
            logger.exception("relevance_filter.stage2_social_batch_failed", source=source)
            if settings.INGEST_SOCIAL_LLM_FAIL_OPEN:
                kept_items.extend(batch)

    logger.info(
        "relevance_filter.stage2_social",
        source=source,
        kept=len(kept_items),
        total=len(capped),
    )
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

        source_url = str(metadata.get("open_access_url") or metadata.get("pdf_url") or metadata.get("url") or resolved.resolved_url)
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
        await redis.enqueue_job("ingest_source_asset", asset_id, _job_id=f"fulltext:{asset_id}")
        scheduled += 1
    return scheduled


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

_SOCIAL_BROAD_TERMS: set[str] = {
    "beauty",
    "skincare",
    "makeup",
    "cosmetics",
    "fragrance",
    "viral",
    "trend",
    "trends",
    "news",
}

_PROJECT_TEXT_STOPWORDS: set[str] = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "about",
    "project",
    "research",
    "analysis",
    "find",
    "highly",
    "relevant",
    "social",
    "track",
    "conversation",
    "conversations",
    "evidence",
    "quality",
    "reports",
    "guidance",
    "projects",
    "latest",
    "recent",
    "new",
    "using",
    "based",
    "youtube",
    "reddit",
    "twitter",
    "tiktok",
    "instagram",
    "platform",
    "platforms",
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
            terms.add(tok)
    return terms


def _match_count(text: str, terms: set[str]) -> int:
    """Count distinct relevance-term matches in text."""
    if not text or not terms:
        return 0
    tokens = _tokenize(text)
    return len(tokens & terms)


def _content_quality_score(title: str, content: str) -> float:
    """Estimate content quality from richness and specificity signals."""
    merged = f"{title} {content}".strip()
    if not merged:
        return 0.0
    tokens = _tokenize(merged)
    unique_token_score = min(1.0, len(tokens) / 80.0)
    length_score = min(1.0, len(content) / 900.0)
    punctuation_count = content.count(".") + content.count("!") + content.count("?")
    structure_score = min(1.0, punctuation_count / 8.0)
    return (unique_token_score * 0.5) + (length_score * 0.35) + (structure_score * 0.15)


def _required_social_match_count(anchor_terms: set[str]) -> int:
    """Use OR semantics: one topical match is enough to keep a candidate."""
    _ = anchor_terms
    return 1


def _required_must_match_count(must_terms: list[str] | set[str] | None = None) -> int:
    """Use OR semantics for explicit must-match terms across all sources."""
    _ = must_terms
    return 1


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


def _normalize_hashtag_token(raw: str) -> str:
    """Normalize free text into a hashtag-safe token without spaces."""
    cleaned = _clean_instagram_keyword_query(raw)
    if not cleaned:
        return ""
    compact = re.sub(r"\s+", "", cleaned).strip().lower()
    return compact if len(compact) >= 3 else ""


def _extract_hashtags(text: str) -> list[str]:
    """Extract cleaned hashtag terms from a social filter string."""
    if not text:
        return []
    tags = re.findall(r"#([a-zA-Z0-9_]+)", text)
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = _normalize_hashtag_token(tag.replace("_", " "))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _social_must_terms(intent: dict, query_terms: list[str]) -> list[str]:
    """Build strict social must-match terms from intent and query terms."""
    explicit = _clean_term_values(intent.get("must_match_terms") or [])
    if explicit:
        return explicit[:5]

    specificity = str(intent.get("query_specificity") or "").strip().lower()
    if specificity != "specific":
        return []

    candidates: list[str] = []
    for value in intent.get("entities") or []:
        candidates.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    for value in intent.get("keywords") or []:
        candidates.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    candidates.extend(query_terms)

    out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        term = str(raw or "").strip().lower().replace("_", " ")
        term = re.sub(r"\s+", " ", term).strip()
        if len(term) < 3:
            continue
        if term in seen:
            continue
        if term in _GENERIC_RELEVANCE_TERMS or term in _SOCIAL_BROAD_TERMS:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= 5:
            break
    return out


def _project_anchor_terms(
    intent: dict,
    *,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> set[str]:
    """Build strict topical anchors from intent + project text for social relevance."""
    raw_terms: list[str] = []
    for value in (
        *(intent.get("must_match_terms") or []),
        *(intent.get("entities") or []),
        *(intent.get("keywords") or []),
        *(intent.get("domain_terms") or []),
        social_filter,
        project_title,
        project_description,
    ):
        raw_terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "").replace("#", " ")))

    anchors: set[str] = set()
    for raw in raw_terms:
        token = str(raw or "").strip().lower().replace("_", " ")
        token = re.sub(r"\s+", " ", token).strip()
        if len(token) < 3:
            continue
        if token in _GENERIC_RELEVANCE_TERMS or token in _SOCIAL_BROAD_TERMS:
            continue
        if token in _PROJECT_TEXT_STOPWORDS:
            continue
        anchors.add(token)
    return anchors


def _instagram_handle_candidates(social_filter: str, intent: dict) -> list[str]:
    """Build candidate Instagram handle queries for fallback account lookup."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _push_candidate(value: str) -> None:
        cleaned_value = str(value or "").strip()
        if not cleaned_value:
            return
        key = cleaned_value.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(cleaned_value)
        if len(candidates) >= max(1, settings.INGEST_INSTAGRAM_ACCOUNT_CANDIDATES):
            return

    explicit_handles = re.findall(r"@([A-Za-z0-9_.]{2,30})", str(social_filter or ""))
    for handle in explicit_handles:
        _push_candidate(handle)
        if len(candidates) >= max(1, settings.INGEST_INSTAGRAM_ACCOUNT_CANDIDATES):
            return candidates

    raw_terms: list[str] = []
    if social_filter:
        raw_terms.append(str(social_filter))
        raw_terms.extend(str(token) for token in str(social_filter).split())
    raw_terms.extend(str(term) for term in (intent.get("must_match_terms") or [])[:8])
    raw_terms.extend(str(term) for term in (intent.get("entities") or [])[:8])
    raw_terms.extend(str(term) for term in (intent.get("keywords") or [])[:8])
    raw_terms.extend(str(term) for term in (intent.get("domain_terms") or [])[:8])

    for raw in raw_terms:
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        cleaned = cleaned.lstrip("#@")
        cleaned = re.sub(r"[^a-zA-Z0-9_.\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        cleaned = _clean_instagram_keyword_query(cleaned)
        if not cleaned:
            continue

        _push_candidate(cleaned)
        if len(candidates) >= max(1, settings.INGEST_INSTAGRAM_ACCOUNT_CANDIDATES):
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
    except asyncio.CancelledError:
        finished_at = datetime.now(UTC).isoformat()
        await _set_ingest_status(
            redis,
            project_id,
            status="failed",
            message="Ingestion cancelled due to worker timeout.",
            queued_at=queued_at,
            started_at=started_at,
            finished_at=finished_at,
            updated_at=finished_at,
        )
        raise
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
        source_diagnostics=outcome.get("source_diagnostics") or {},
        fulltext_enqueued=int(outcome.get("fulltext_enqueued") or 0),
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
                   tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled,
                   papers_enabled,
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
    youtube_enabled = bool(project.get("youtube_enabled"))
    reddit_enabled = bool(project.get("reddit_enabled"))
    x_enabled = bool(project.get("x_enabled"))
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
            note="Social sources remain enabled and use intent/context-derived queries.",
        )

    strict_social_terms = sorted(
        _project_anchor_terms(
            intent,
            social_filter=social_filter,
            project_title=project_title,
            project_description=project_description,
        )
    )

    logger.info(
        "ingest_project.start",
        project_id=project_id,
        social_filter=social_filter[:60],
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

    source_types = [
        "social_tiktok",
        "social_instagram",
        "social_youtube",
        "social_reddit",
        "social_x",
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

    gather_results = await asyncio.gather(
        _run_source_with_timeout(
            "social_tiktok",
            _ingest_tiktok(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
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

    expansion_meta: dict[str, dict[str, int | bool | str]] = {}
    mutable_results = list(gather_results)
    social_index_map = {
        "social_tiktok": 0,
        "social_instagram": 1,
        "social_youtube": 2,
        "social_reddit": 3,
        "social_x": 4,
    }
    social_enabled_map = {
        "social_tiktok": tiktok_enabled,
        "social_instagram": instagram_enabled,
        "social_youtube": youtube_enabled,
        "social_reddit": reddit_enabled,
        "social_x": x_enabled,
    }

    raw_social_fast_total = 0
    for source_name, idx in social_index_map.items():
        result = mutable_results[idx]
        fast_count = len(result) if isinstance(result, list) else 0
        raw_social_fast_total += fast_count
        expansion_meta[source_name] = {
            "raw_fast": fast_count,
            "raw_expanded": 0,
            "raw_final": fast_count,
            "expansion_triggered": False,
            "expansion_reason": "",
        }

    if settings.INGEST_SOCIAL_EXPANSION_ENABLED:
        expanded_social_filter = _build_expanded_social_filter(
            intent,
            social_filter=social_filter,
            project_title=project_title,
            project_description=project_description,
        )
        for source_name, idx in social_index_map.items():
            if not social_enabled_map[source_name]:
                continue
            source_result = mutable_results[idx]
            source_fast = len(source_result) if isinstance(source_result, list) else 0
            source_min = (
                int(settings.INGEST_SOCIAL_RAW_MIN_PER_WEAK_SOURCE)
                if source_name in {"social_instagram", "social_youtube"}
                else int(settings.INGEST_SOCIAL_RAW_MIN_PER_SOURCE)
            )
            needs_expansion = (
                source_fast < source_min
                or raw_social_fast_total < int(settings.INGEST_SOCIAL_RAW_TARGET_TOTAL)
            )
            if not needs_expansion:
                continue

            reason = (
                "low_source_and_low_total"
                if source_fast < source_min
                and raw_social_fast_total < int(settings.INGEST_SOCIAL_RAW_TARGET_TOTAL)
                else ("low_source_raw" if source_fast < source_min else "low_total_raw")
            )
            expansion_meta[source_name]["expansion_triggered"] = True
            expansion_meta[source_name]["expansion_reason"] = reason
            try:
                if source_name == "social_tiktok":
                    expanded_docs = await _run_source_with_timeout(
                        "social_tiktok_expand",
                        _ingest_tiktok(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=expanded_social_filter,
                            project_title=project_title,
                            project_description=project_description,
                            max_results_override=settings.INGEST_SOCIAL_EXPANDED_TIKTOK_MAX_RESULTS,
                        ),
                        timeout_seconds=settings.INGEST_SOCIAL_EXPANSION_TIMEOUT,
                    )
                elif source_name == "social_instagram":
                    expanded_docs = await _run_source_with_timeout(
                        "social_instagram_expand",
                        _ingest_instagram(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=expanded_social_filter,
                            project_title=project_title,
                            project_description=project_description,
                            oldest_timestamp=oldest_timestamp,
                            redis=redis,
                            max_pages_override=settings.INGEST_SOCIAL_EXPANDED_INSTAGRAM_HASHTAG_PAGES,
                            max_age_days_override=settings.INGEST_SOCIAL_EXPANDED_MAX_AGE_DAYS,
                        ),
                        timeout_seconds=settings.INGEST_SOCIAL_EXPANSION_TIMEOUT,
                    )
                elif source_name == "social_youtube":
                    expanded_docs = await _run_source_with_timeout(
                        "social_youtube_expand",
                        _ingest_youtube(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=expanded_social_filter,
                            project_title=project_title,
                            project_description=project_description,
                            fetch_limit_override=settings.INGEST_SOCIAL_EXPANDED_FETCH_LIMIT_PER_SOURCE,
                            expansion_mode=True,
                        ),
                        timeout_seconds=settings.INGEST_SOCIAL_EXPANSION_TIMEOUT,
                    )
                elif source_name == "social_reddit":
                    expanded_docs = await _run_source_with_timeout(
                        "social_reddit_expand",
                        _ingest_reddit(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=expanded_social_filter,
                            project_title=project_title,
                            project_description=project_description,
                            fetch_limit_override=settings.INGEST_SOCIAL_EXPANDED_FETCH_LIMIT_PER_SOURCE,
                            expansion_mode=True,
                        ),
                        timeout_seconds=settings.INGEST_SOCIAL_EXPANSION_TIMEOUT,
                    )
                else:
                    expanded_docs = await _run_source_with_timeout(
                        "social_x_expand",
                        _ingest_x(
                            project_id=project_id,
                            user_id=user_id,
                            intent=intent,
                            social_filter=expanded_social_filter,
                            project_title=project_title,
                            project_description=project_description,
                            fetch_limit_override=settings.INGEST_SOCIAL_EXPANDED_FETCH_LIMIT_PER_SOURCE,
                            expansion_mode=True,
                        ),
                        timeout_seconds=settings.INGEST_SOCIAL_EXPANSION_TIMEOUT,
                    )
            except Exception:
                logger.exception("ingest_project.social_expansion_failed", project_id=project_id, source=source_name)
                expanded_docs = []

            merged = _dedupe_social_docs(
                list(source_result if isinstance(source_result, list) else []) + list(expanded_docs or [])
            )
            mutable_results[idx] = merged
            expanded_count = len(expanded_docs or [])
            expansion_meta[source_name]["raw_expanded"] = expanded_count
            expansion_meta[source_name]["raw_final"] = len(merged)
            raw_social_fast_total += max(0, len(merged) - source_fast)
            logger.info(
                "ingest_project.social_expansion",
                project_id=project_id,
                source=source_name,
                raw_fast=source_fast,
                raw_expanded=expanded_count,
                raw_final=len(merged),
                reason=reason,
            )

    intent_text = json.dumps(intent)
    documents: list[RawDocument] = []
    any_source_completed = False
    source_diagnostics: dict[str, dict] = {}

    for result, source, source_enabled in zip(
        mutable_results, source_types, source_enabled_flags, strict=False
    ):
        if not source_enabled:
            continue
        if isinstance(result, BaseException):
            logger.warning(
                "ingest_tool.failed", project_id=project_id, source=source, error=str(result)
            )
            source_diagnostics[source] = {
                "fetched": 0,
                "kept": 0,
                "filtered_out": 0,
                "reason": "source_failed",
            }
            source_diagnostics[source].update(expansion_meta.get(source, {}))
            continue
        any_source_completed = True
        if not result:
            source_diagnostics[source] = {
                "fetched": 0,
                "kept": 0,
                "filtered_out": 0,
                "reason": "no_results_from_source",
            }
            source_diagnostics[source].update(expansion_meta.get(source, {}))
            continue

        if oldest_timestamp is not None:
            result = [doc for doc in result if _is_document_new_enough(doc, oldest_timestamp)]
            if not result:
                source_diagnostics[source] = {
                    "fetched": 0,
                    "kept": 0,
                    "filtered_out": 0,
                    "reason": "all_older_than_refresh_watermark",
                }
                source_diagnostics[source].update(expansion_meta.get(source, {}))
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

        fetched_count = len(result)
        social_terms_for_source = strict_social_terms if source in SOCIAL_SOURCES else []
        filtered_items = await _filter_relevance(
            filter_items,
            intent_text,
            source,
            redis,
            must_match_terms=social_terms_for_source if source in SOCIAL_SOURCES else intent.get("must_match_terms") or [],
            social_match_terms=social_terms_for_source,
        )
        kept_indexes = {
            int(item.get("_idx", -1))
            for item in filtered_items
            if isinstance(item, dict) and isinstance(item.get("_idx"), int)
        }
        kept_count = len(kept_indexes)
        source_diagnostics[source] = {
            "fetched": fetched_count,
            "kept": kept_count,
            "filtered_out": max(0, fetched_count - kept_count),
            "reason": (
                "strict_relevance_filtered"
                if source in SOCIAL_SOURCES and kept_count == 0 and fetched_count > 0
                else "ok"
            ),
        }
        source_diagnostics[source].update(expansion_meta.get(source, {}))
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
                "message": "Ingestion completed but no new documents passed strict relevance filters.",
                "source_counts": {},
                "source_diagnostics": source_diagnostics,
                "total_chunks": int(total or 0),
            }
        return {
            "status": "failed",
            "message": "No source completed successfully.",
            "source_counts": {},
            "source_diagnostics": source_diagnostics,
            "total_chunks": 0,
        }

    inserted = await ingest_documents(documents)
    fulltext_enqueued = await _schedule_fulltext_enrichment(ctx, pool, documents)

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
        social_youtube_docs=sum(1 for d in documents if d.source == "social_youtube"),
        social_reddit_docs=sum(1 for d in documents if d.source == "social_reddit"),
        social_x_docs=sum(1 for d in documents if d.source == "social_x"),
        paper_docs=sum(1 for d in documents if d.source == "paper"),
        patent_docs=sum(1 for d in documents if d.source == "patent"),
        news_docs=sum(1 for d in documents if d.source == "news"),
        search_docs=sum(1 for d in documents if d.source == "search"),
        total_docs=len(documents),
        fulltext_enqueued=fulltext_enqueued,
        chunks_inserted=inserted,
        total_chunks=total,
    )
    source_counts = {
        "social_instagram": sum(1 for d in documents if d.source == "social_instagram"),
        "social_tiktok": sum(1 for d in documents if d.source == "social_tiktok"),
        "social_youtube": sum(1 for d in documents if d.source == "social_youtube"),
        "social_reddit": sum(1 for d in documents if d.source == "social_reddit"),
        "social_x": sum(1 for d in documents if d.source == "social_x"),
        "paper": sum(1 for d in documents if d.source == "paper"),
        "patent": sum(1 for d in documents if d.source == "patent"),
        "news": sum(1 for d in documents if d.source == "news"),
        "search": sum(1 for d in documents if d.source == "search"),
    }
    social_total = (
        source_counts["social_instagram"]
        + source_counts["social_tiktok"]
        + source_counts["social_youtube"]
        + source_counts["social_reddit"]
        + source_counts["social_x"]
    )
    social_enabled = any([tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled])
    message = "Ingestion completed."
    if social_enabled and social_total == 0:
        message = "Ingestion completed. Insufficient relevant social content for strict mode."
    return {
        "status": "ready",
        "message": message,
        "source_counts": source_counts,
        "source_diagnostics": source_diagnostics,
        "fulltext_enqueued": fulltext_enqueued,
        "total_chunks": int(total or 0),
    }


async def _ingest_instagram(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
    oldest_timestamp: int | None,
    redis,
    max_pages_override: int | None = None,
    max_age_days_override: int | None = None,
) -> list[RawDocument]:
    """Ingest Instagram reels via hashtag-first discovery, fail-open on errors."""
    _ = redis
    fallback_query = _query_for_social(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    handle_candidates = _instagram_handle_candidates(social_filter, intent)
    hashtags = _extract_hashtags(social_filter)
    if not hashtags:
        for must in intent.get("must_match_terms") or []:
            cleaned = _normalize_hashtag_token(str(must))
            if cleaned:
                hashtags.append(cleaned)
        for keyword in intent.get("keywords") or []:
            cleaned = _normalize_hashtag_token(str(keyword))
            if cleaned:
                hashtags.append(cleaned)
            if len(hashtags) >= settings.INGEST_INSTAGRAM_HASHTAG_QUERIES:
                break
        for entity in intent.get("entities") or []:
            cleaned = _normalize_hashtag_token(str(entity))
            if cleaned:
                hashtags.append(cleaned)
            if len(hashtags) >= settings.INGEST_INSTAGRAM_HASHTAG_QUERIES:
                break
        hashtags = list(dict.fromkeys(hashtags))[: settings.INGEST_INSTAGRAM_HASHTAG_QUERIES]
    if not hashtags:
        fallback_reels = await _fetch_instagram_reels_from_candidates(
            project_id=project_id,
            candidate_queries=handle_candidates,
            oldest_timestamp=oldest_timestamp,
        )
        if not fallback_reels:
            logger.warning("ingest_instagram.no_hashtags", project_id=project_id)
            return await _social_web_fallback_docs(
                source="social_instagram",
                project_id=project_id,
                user_id=user_id,
                query=fallback_query,
                keep_limit=max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE),
            )
        all_posts: list[dict] = fallback_reels
    else:
        all_posts = []

    logger.info("ingest_instagram.start", project_id=project_id, hashtag_count=len(hashtags))

    now_utc = datetime.now(UTC)
    cutoff_days = (
        max(1, int(max_age_days_override))
        if isinstance(max_age_days_override, int) and max_age_days_override > 0
        else max(1, int(settings.INGEST_SOCIAL_MAX_AGE_DAYS))
    )
    cutoff_window = now_utc - timedelta(days=cutoff_days)
    oldest_cutoff = (
        datetime.fromtimestamp(oldest_timestamp, tz=UTC)
        if isinstance(oldest_timestamp, int | float) and oldest_timestamp > 0
        else None
    )
    recency_cutoff = max(cutoff_window, oldest_cutoff) if oldest_cutoff else cutoff_window

    if hashtags:
        max_pages = (
            max(1, int(max_pages_override))
            if isinstance(max_pages_override, int) and max_pages_override > 0
            else max(1, settings.INGEST_INSTAGRAM_HASHTAG_PAGES)
        )
        for hashtag in hashtags[: max(1, settings.INGEST_INSTAGRAM_HASHTAG_QUERIES)]:
            cursor: str | None = None
            empty_page_streak = 0
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

                if not posts and next_cursor:
                    empty_page_streak += 1
                    logger.info(
                        "ingest_instagram.empty_page_break",
                        project_id=project_id,
                        hashtag=hashtag,
                        has_cursor=bool(cursor),
                    )
                    if empty_page_streak >= 1:
                        break
                else:
                    empty_page_streak = 0

                for post in posts:
                    ts = post.get("timestamp")
                    if isinstance(ts, int | float):
                        post_dt = datetime.fromtimestamp(float(ts), tz=UTC)
                        if post_dt < recency_cutoff:
                            continue
                    all_posts.append(post)

                if next_cursor == cursor:
                    break
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
            return await _social_web_fallback_docs(
                source="social_instagram",
                project_id=project_id,
                user_id=user_id,
                query=fallback_query,
                keep_limit=max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE),
            )
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
    )[: max(1, settings.INGEST_INSTAGRAM_ACCOUNTS_TO_FETCH)]
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
    anchor_terms = _project_anchor_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    min_match = max(1, int(settings.INGEST_SOCIAL_MIN_RELEVANCE_MATCHES))

    scored: list[tuple[int, float, float, float, dict]] = []
    for reel in all_reels:
        caption = str(reel.get("caption") or "").strip()
        if not caption:
            continue

        likes = _as_int(reel.get("like_count"))
        views = _as_int(reel.get("view_count") or reel.get("play_count"))
        relevance_match = int(
            _match_count(caption, anchor_terms)
            + _match_count(caption, relevance_terms)
            + _match_count(caption, brand_terms)
        )
        if relevance_match < min_match:
            continue
        quality_score = _content_quality_score(str(reel.get("username") or ""), caption)
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
        scored.append((relevance_match, quality_score, engagement_score, recency_score, reel))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    docs: list[RawDocument] = []
    seen_ids: set[str] = set()
    limit = max(
        1,
        min(settings.INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE),
    )
    for _, _, _, _, reel in scored:
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
    if not docs:
        return await _social_web_fallback_docs(
            source="social_instagram",
            project_id=project_id,
            user_id=user_id,
            query=fallback_query,
            keep_limit=max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE),
        )
    return docs


async def _ingest_tiktok(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
    max_results_override: int | None = None,
) -> list[RawDocument]:
    """
    Fetch recent TikTok posts for the handle derived from social_filter.

    social_filter is reused as the TikTok handle (same structured_intent field).
    Videos with an empty description are skipped — they carry no KB value.

    Returns a list of RawDocument with source="social_tiktok".
    Returns [] (and logs a warning) if the fetch yields no posts.
    Never raises — all exceptions are caught and logged.
    """
    fallback_query = _query_for_social(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    query_terms = _social_query_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query_terms:
        logger.warning("ingest_tiktok.no_query_terms", project_id=project_id)
        return await _social_web_fallback_docs(
            source="social_tiktok",
            project_id=project_id,
            user_id=user_id,
            query=fallback_query,
            keep_limit=max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE),
        )

    specificity = str(intent.get("query_specificity") or "").strip().lower()
    must_terms = _social_must_terms(intent, query_terms)
    anchor_terms = _project_anchor_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    exact_match = specificity == "specific" or bool(must_terms)
    keyword_queries = query_terms[: max(1, settings.INGEST_TIKTOK_KEYWORD_LIMIT)]
    max_results = (
        max(1, int(max_results_override))
        if isinstance(max_results_override, int) and max_results_override > 0
        else settings.INGEST_TIKTOK_MAX_RESULTS
    )
    hashtag_query = " ".join(
        f"#{tag}" for tag in (_normalize_hashtag_token(term) for term in keyword_queries) if tag
    )

    posts = await fetch_tiktok_posts(
        hashtags=hashtag_query,
        keyword_queries=keyword_queries,
        exact_match=exact_match,
        period="90" if exact_match else "30",
        max_results=max_results,
    )
    if len(posts) < settings.INGEST_TIKTOK_MIN_RELEVANT_RESULTS and exact_match:
        # Controlled expansion: relax exact matching but keep the same topical query terms.
        expanded = await fetch_tiktok_posts(
            hashtags=hashtag_query,
            keyword_queries=keyword_queries,
            exact_match=False,
            period="30",
            max_results=max_results,
        )
        seen_ids = {str(item.get("video_id") or "") for item in posts}
        for item in expanded:
            vid = str(item.get("video_id") or "")
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)
            posts.append(item)
            if len(posts) >= max_results:
                break

    logger.info(
        "ingest_project.tiktok.posts_fetched",
        project_id=project_id,
        exact_match=exact_match,
        query_terms=keyword_queries[:6],
        post_count=len(posts),
    )

    scored: list[tuple[int, float, float, float, dict]] = []
    skipped_non_match = 0
    required_must_hits = _required_must_match_count(must_terms)
    min_match = max(
        int(settings.INGEST_SOCIAL_MIN_RELEVANCE_MATCHES),
        _required_social_match_count(anchor_terms),
    )
    now_utc = datetime.now(UTC)
    for video in posts:
        description = (video.get("description") or "").strip()
        if not description:
            continue
        if must_terms and _match_count(description, set(_clean_term_values(must_terms))) < required_must_hits:
            skipped_non_match += 1
            continue
        relevance_match = int(_match_count(description, anchor_terms))
        if relevance_match < min_match:
            skipped_non_match += 1
            continue
        quality_score = _content_quality_score(str(video.get("author_username") or ""), description)
        engagement_score = min(
            1.0,
            math.log1p(max(0, _as_int(video.get("likes")) + _as_int(video.get("views")))) / 20.0,
        )
        recency_score = 0.4
        ts = video.get("create_time") or video.get("created_at")
        if ts:
            try:
                if isinstance(ts, int | float) or str(ts).isdigit():
                    video_dt = datetime.fromtimestamp(float(ts), tz=UTC)
                else:
                    normalized = str(ts).replace("Z", "+00:00")
                    video_dt = datetime.fromisoformat(normalized)
                    if video_dt.tzinfo is None:
                        video_dt = video_dt.replace(tzinfo=UTC)
                days_old = max(0.0, (now_utc - video_dt).total_seconds() / 86400.0)
                recency_score = math.exp(-days_old / 21.0)
            except Exception:
                recency_score = 0.4
        scored.append((relevance_match, quality_score, engagement_score, recency_score, video))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    docs: list[RawDocument] = []
    keep_limit = max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE)
    for _, _, _, _, video in scored:
        description = str(video.get("description") or "").strip()
        if not description:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_tiktok",
                source_id=video.get("video_id", ""),
                title=f"@{video.get('author_username', '')}",
                content=description[:2000],
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
        if len(docs) >= keep_limit:
            break

    logger.info(
        "ingest_project.tiktok.docs_filtered",
        project_id=project_id,
        candidates=len(posts),
        docs=len(docs),
        skipped_non_match=skipped_non_match,
        must_terms=must_terms[:5],
    )
    if not docs:
        return await _social_web_fallback_docs(
            source="social_tiktok",
            project_id=project_id,
            user_id=user_id,
            query=fallback_query,
            keep_limit=max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE),
        )
    return docs


async def _ingest_x(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
    fetch_limit_override: int | None = None,
    expansion_mode: bool = False,
) -> list[RawDocument]:
    """Fetch recent X posts and map them into social KB documents."""
    query = _query_for_social(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_x.no_query", project_id=project_id)
        return []

    default_fetch_limit = max(
        settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE,
        settings.INGEST_SOCIAL_FETCH_LIMIT_PER_SOURCE,
    )
    fetch_limit = (
        max(1, int(fetch_limit_override))
        if isinstance(fetch_limit_override, int) and fetch_limit_override > 0
        else default_fetch_limit
    )
    keep_limit = max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE)

    posts: list[dict] = []
    seen_source_ids: set[str] = set()
    try:
        for query_variant in _social_query_variants(
            intent=intent,
            base_query=query,
            social_filter=social_filter,
            project_title=project_title,
            project_description=project_description,
            expanded=expansion_mode,
        ):
            batch = await search_x_posts(query_variant, max_results=fetch_limit)
            for row in batch:
                source_id = str(row.get("source_id") or row.get("id") or row.get("url") or "").strip()
                if not source_id or source_id in seen_source_ids:
                    continue
                seen_source_ids.add(source_id)
                posts.append(row)
                if len(posts) >= fetch_limit:
                    break
            if len(posts) >= fetch_limit:
                break
    except Exception:
        logger.exception("ingest_x.failed", project_id=project_id)
        return await _social_web_fallback_docs(
            source="social_x",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )
    if not posts:
        return await _social_web_fallback_docs(
            source="social_x",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )

    relevance_terms = _build_relevance_terms(intent, query)
    brand_terms = _build_brand_terms(intent)
    anchor_terms = _project_anchor_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    min_match = max(
        int(settings.INGEST_SOCIAL_MIN_RELEVANCE_MATCHES),
        _required_social_match_count(anchor_terms),
    )
    now_utc = datetime.now(UTC)
    scored: list[tuple[int, float, float, float, dict]] = []
    for post in posts:
        content = str(post.get("content") or "").strip()
        if not content:
            continue
        likes = _as_int(post.get("likes"))
        retweets = _as_int(post.get("retweets"))
        replies = _as_int(post.get("replies"))
        relevance_match = int(
            _match_count(content, anchor_terms)
            + _match_count(content, relevance_terms)
            + _match_count(content, brand_terms)
        )
        if relevance_match < min_match:
            continue
        quality_score = _content_quality_score(str(post.get("author") or ""), content)
        engagement_score = min(1.0, math.log1p(max(0, likes + retweets + replies)) / 20.0)

        recency_score = 0.4
        published_raw = post.get("published_at")
        if published_raw:
            try:
                normalized = str(published_raw).replace("Z", "+00:00")
                published_dt = datetime.fromisoformat(normalized)
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=UTC)
                days_old = max(0.0, (now_utc - published_dt).total_seconds() / 86400.0)
                recency_score = math.exp(-days_old / 14.0)
            except Exception:
                recency_score = 0.4

        scored.append((relevance_match, quality_score, engagement_score, recency_score, post))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    docs: list[RawDocument] = []
    for _, _, _, _, post in scored:
        content = str(post.get("content") or "").strip()
        source_id = str(post.get("source_id") or "").strip()
        if not content or not source_id:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_x",
                source_id=source_id,
                title=f"@{str(post.get('author') or 'x')}",
                content=content[:2000],
                metadata={
                    "platform": "x",
                    "author": post.get("author") or "",
                    "likes": _as_int(post.get("likes")),
                    "retweets": _as_int(post.get("retweets")),
                    "replies": _as_int(post.get("replies")),
                    "quotes": _as_int(post.get("quotes")),
                    "published_at": post.get("published_at") or "",
                    "url": post.get("url") or "",
                    "tool": "ensemble_twitter",
                },
            )
        )
        if len(docs) >= keep_limit:
            break

    logger.info("ingest_x.success", project_id=project_id, count=len(docs))
    if not docs:
        return await _social_web_fallback_docs(
            source="social_x",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )
    return docs


async def _ingest_reddit(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
    fetch_limit_override: int | None = None,
    expansion_mode: bool = False,
) -> list[RawDocument]:
    """Fetch Reddit posts and map them into social KB documents."""
    query = _query_for_social(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    query = re.sub(r"@[A-Za-z0-9_]{2,20}", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        logger.warning("ingest_reddit.no_query", project_id=project_id)
        return []

    default_fetch_limit = max(
        settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE,
        settings.INGEST_SOCIAL_FETCH_LIMIT_PER_SOURCE,
    )
    fetch_limit = (
        max(1, int(fetch_limit_override))
        if isinstance(fetch_limit_override, int) and fetch_limit_override > 0
        else default_fetch_limit
    )
    keep_limit = max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE)

    posts: list[dict] = []
    seen_source_ids: set[str] = set()
    try:
        for query_variant in _social_query_variants(
            intent=intent,
            base_query=query,
            social_filter=social_filter,
            project_title=project_title,
            project_description=project_description,
            expanded=expansion_mode,
        ):
            batch = await search_reddit_posts(query_variant, limit=fetch_limit)
            for row in batch:
                source_id = str(row.get("source_id") or row.get("id") or row.get("url") or "").strip()
                if not source_id or source_id in seen_source_ids:
                    continue
                seen_source_ids.add(source_id)
                posts.append(row)
                if len(posts) >= fetch_limit:
                    break
            if len(posts) >= fetch_limit:
                break
    except Exception:
        logger.exception("ingest_reddit.failed", project_id=project_id)
        return await _social_web_fallback_docs(
            source="social_reddit",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )
    if not posts:
        return await _social_web_fallback_docs(
            source="social_reddit",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )

    relevance_terms = _build_relevance_terms(intent, query)
    brand_terms = _build_brand_terms(intent)
    anchor_terms = _project_anchor_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    min_match = max(
        int(settings.INGEST_SOCIAL_MIN_RELEVANCE_MATCHES),
        _required_social_match_count(anchor_terms),
    )
    now_utc = datetime.now(UTC)
    scored: list[tuple[int, float, float, float, dict]] = []
    for post in posts:
        source_id = str(post.get("source_id") or "").strip()
        title = str(post.get("title") or "").strip()
        body = str(post.get("content") or "").strip()
        content = f"{title}\n\n{body}".strip()
        if not source_id or not content:
            continue
        score_count = _as_int(post.get("score"))
        comments_count = _as_int(post.get("comments"))
        relevance_match = int(
            _match_count(content, anchor_terms)
            + _match_count(content, relevance_terms)
            + _match_count(content, brand_terms)
        )
        if relevance_match < min_match:
            continue
        quality_score = _content_quality_score(title, content)
        engagement_score = min(1.0, math.log1p(max(0, score_count + comments_count)) / 20.0)

        recency_score = 0.4
        published_raw = post.get("published_at")
        if published_raw:
            try:
                if isinstance(published_raw, int | float) or str(published_raw).isdigit():
                    published_dt = datetime.fromtimestamp(float(published_raw), tz=UTC)
                else:
                    normalized = str(published_raw).replace("Z", "+00:00")
                    published_dt = datetime.fromisoformat(normalized)
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=UTC)
                days_old = max(0.0, (now_utc - published_dt).total_seconds() / 86400.0)
                recency_score = math.exp(-days_old / 21.0)
            except Exception:
                recency_score = 0.4

        scored.append((relevance_match, quality_score, engagement_score, recency_score, post))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    docs: list[RawDocument] = []
    for _, _, _, _, post in scored:
        source_id = str(post.get("source_id") or "").strip()
        title = str(post.get("title") or "").strip()
        body = str(post.get("content") or "").strip()
        content = f"{title}\n\n{body}".strip()
        if not source_id or not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_reddit",
                source_id=source_id,
                title=title or "Reddit post",
                content=content[:2000],
                metadata={
                    "platform": "reddit",
                    "author": post.get("author") or "",
                    "subreddit": post.get("subreddit") or "",
                    "score": _as_int(post.get("score")),
                    "comments": _as_int(post.get("comments")),
                    "published_at": post.get("published_at") or "",
                    "url": post.get("url") or "",
                    "tool": "ensemble_reddit",
                },
            )
        )
        if len(docs) >= keep_limit:
            break

    logger.info("ingest_reddit.success", project_id=project_id, count=len(docs))
    if not docs:
        return await _social_web_fallback_docs(
            source="social_reddit",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )
    return docs


async def _ingest_youtube(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
    fetch_limit_override: int | None = None,
    expansion_mode: bool = False,
) -> list[RawDocument]:
    """Fetch YouTube videos and map them into social KB documents."""
    query = _query_for_social(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    query = re.sub(r"@[A-Za-z0-9_]{2,20}", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        logger.warning("ingest_youtube.no_query", project_id=project_id)
        return []

    default_fetch_limit = max(
        settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE,
        settings.INGEST_SOCIAL_FETCH_LIMIT_PER_SOURCE,
    )
    fetch_limit = (
        max(1, int(fetch_limit_override))
        if isinstance(fetch_limit_override, int) and fetch_limit_override > 0
        else default_fetch_limit
    )
    keep_limit = max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE)

    videos: list[dict] = []
    seen_source_ids: set[str] = set()
    try:
        for query_variant in _social_query_variants(
            intent=intent,
            base_query=query,
            social_filter=social_filter,
            project_title=project_title,
            project_description=project_description,
            expanded=expansion_mode,
        ):
            batch = await search_youtube_videos(query_variant, max_results=fetch_limit)
            for row in batch:
                source_id = str(row.get("source_id") or row.get("video_id") or row.get("url") or "").strip()
                if not source_id or source_id in seen_source_ids:
                    continue
                seen_source_ids.add(source_id)
                videos.append(row)
                if len(videos) >= fetch_limit:
                    break
            if len(videos) >= fetch_limit:
                break
    except Exception:
        logger.exception("ingest_youtube.failed", project_id=project_id)
        return await _social_web_fallback_docs(
            source="social_youtube",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )
    if not videos:
        return await _social_web_fallback_docs(
            source="social_youtube",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
        )

    relevance_terms = _build_relevance_terms(intent, query)
    brand_terms = _build_brand_terms(intent)
    anchor_terms = _project_anchor_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    min_match = max(
        int(settings.INGEST_SOCIAL_MIN_RELEVANCE_MATCHES),
        _required_social_match_count(anchor_terms),
    )
    now_utc = datetime.now(UTC)
    scored: list[tuple[int, float, float, float, dict]] = []
    for video in videos:
        title = str(video.get("title") or "").strip()
        content = str(video.get("content") or "").strip()
        if not content:
            continue
        likes = _as_int(video.get("likes"))
        views = _as_int(video.get("views"))
        comments = _as_int(video.get("comments"))
        merged_text = f"{title} {content}"
        relevance_match = int(
            _match_count(merged_text, anchor_terms)
            + _match_count(merged_text, relevance_terms)
            + _match_count(merged_text, brand_terms)
        )
        if relevance_match < min_match:
            continue
        quality_score = _content_quality_score(title, content)
        engagement_score = min(1.0, math.log1p(max(0, likes + views + comments)) / 20.0)

        recency_score = 0.4
        published_raw = video.get("published_at")
        if published_raw:
            try:
                normalized = str(published_raw).replace("Z", "+00:00")
                published_dt = datetime.fromisoformat(normalized)
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=UTC)
                days_old = max(0.0, (now_utc - published_dt).total_seconds() / 86400.0)
                recency_score = math.exp(-days_old / 21.0)
            except Exception:
                recency_score = 0.4

        scored.append((relevance_match, quality_score, engagement_score, recency_score, video))

    if not scored and videos:
        logger.info("ingest_youtube.relevance_fallback", project_id=project_id, candidate_count=len(videos))
        for video in videos:
            title = str(video.get("title") or "").strip()
            content = str(video.get("content") or "").strip()
            if not content:
                continue
            merged_text = f"{title} {content}"
            lexical_match = int(
                _match_count(merged_text, anchor_terms)
                + _match_count(merged_text, relevance_terms)
                + _match_count(merged_text, brand_terms)
            )
            if lexical_match < 1:
                continue
            likes = _as_int(video.get("likes"))
            views = _as_int(video.get("views"))
            comments = _as_int(video.get("comments"))
            quality_score = _content_quality_score(title, content)
            engagement_score = min(1.0, math.log1p(max(0, likes + views + comments)) / 20.0)
            scored.append((0, quality_score, engagement_score, 0.4, video))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    docs: list[RawDocument] = []
    for _, _, _, _, video in scored:
        source_id = str(video.get("source_id") or "").strip()
        title = str(video.get("title") or "").strip()
        content = str(video.get("content") or "").strip()
        if not source_id or not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_youtube",
                source_id=source_id,
                title=title or "YouTube video",
                content=content[:2000],
                metadata={
                    "platform": "youtube",
                    "author": video.get("author") or "",
                    "thumbnail_url": video.get("thumbnail_url") or "",
                    "cover_url": video.get("thumbnail_url") or "",
                    "likes": _as_int(video.get("likes")),
                    "views": _as_int(video.get("views")),
                    "comments": _as_int(video.get("comments")),
                    "published_at": video.get("published_at") or "",
                    "url": video.get("url") or "",
                    "tool": "ensemble_youtube",
                },
            )
        )
        if len(docs) >= keep_limit:
            break

    logger.info("ingest_youtube.success", project_id=project_id, count=len(docs))
    if not docs:
        return await _social_web_fallback_docs(
            source="social_youtube",
            project_id=project_id,
            user_id=user_id,
            query=query,
            keep_limit=keep_limit,
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

    papers: list[dict] = []
    seen_papers: set[str] = set()
    for query_variant in _query_variants_for_source(intent, query):
        batch = await fetch_papers(query_variant)
        for paper in batch:
            paper_key = str(
                paper.get("paper_id") or paper.get("doi") or paper.get("title") or ""
            ).strip()
            if not paper_key:
                continue
            normalized_key = paper_key.lower()
            if normalized_key in seen_papers:
                continue
            seen_papers.add(normalized_key)
            papers.append(paper)
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
                    "url": paper.get("url") or "",
                    "pdf_url": paper.get("pdf_url") or "",
                    "open_access_url": paper.get("open_access_url") or "",
                    "is_open_access": bool(paper.get("is_open_access") or False),
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
        papers: list[dict] = []
        seen_papers: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_semantic_scholar(
                query_variant,
                must_match_terms=intent.get("must_match_terms") or [],
                domain_terms=intent.get("domain_terms") or [],
                query_specificity=intent.get("query_specificity"),
            )
            for paper in batch:
                paper_key = str(
                    paper.get("paper_id") or paper.get("url") or paper.get("title") or ""
                ).strip()
                if not paper_key:
                    continue
                normalized_key = paper_key.lower()
                if normalized_key in seen_papers:
                    continue
                seen_papers.add(normalized_key)
                papers.append(paper)
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
        papers: list[dict] = []
        seen_papers: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_pubmed(
                query_variant,
                must_match_terms=intent.get("must_match_terms") or [],
                domain_terms=intent.get("domain_terms") or [],
                query_specificity=intent.get("query_specificity"),
            )
            for paper in batch:
                paper_key = str(
                    paper.get("pmid") or paper.get("url") or paper.get("title") or ""
                ).strip()
                if not paper_key:
                    continue
                normalized_key = paper_key.lower()
                if normalized_key in seen_papers:
                    continue
                seen_papers.add(normalized_key)
                papers.append(paper)
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
        papers: list[dict] = []
        seen_papers: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_arxiv(
                query_variant,
                must_match_terms=intent.get("must_match_terms") or [],
                domain_terms=intent.get("domain_terms") or [],
                query_specificity=intent.get("query_specificity"),
            )
            for paper in batch:
                paper_key = str(
                    paper.get("arxiv_id") or paper.get("url") or paper.get("title") or ""
                ).strip()
                if not paper_key:
                    continue
                normalized_key = paper_key.lower()
                if normalized_key in seen_papers:
                    continue
                seen_papers.add(normalized_key)
                papers.append(paper)
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
        patents: list[dict] = []
        seen_patents: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_patentsview(
                query_variant,
                must_match_terms=intent.get("must_match_terms") or [],
                domain_terms=intent.get("domain_terms") or [],
                query_specificity=intent.get("query_specificity"),
            )
            for patent in batch:
                patent_key = str(
                    patent.get("patent_number") or patent.get("url") or patent.get("title") or ""
                ).strip()
                if not patent_key:
                    continue
                normalized_key = patent_key.lower()
                if normalized_key in seen_patents:
                    continue
                seen_patents.add(normalized_key)
                patents.append(patent)
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
                    "fulltext_url": patent.get("url") or "",
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
        patents: list[dict] = []
        seen_patents: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_lens(query_variant)
            for patent in batch:
                patent_key = str(
                    patent.get("lens_id") or patent.get("patent_number") or patent.get("title") or ""
                ).strip()
                if not patent_key:
                    continue
                normalized_key = patent_key.lower()
                if normalized_key in seen_patents:
                    continue
                seen_patents.add(normalized_key)
                patents.append(patent)
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
                    "fulltext_url": patent.get("url") or "",
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
    return str(seed_query or "").strip()


def _project_context_query(project_title: str, project_description: str) -> str:
    """Create a short fallback query from project title/description."""
    title = str(project_title or "").strip()
    description = str(project_description or "").strip()
    if not title and not description:
        return ""
    text = f"{title} {description}".strip()
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return " ".join(tokens[:12]).strip()


def _social_query_terms(
    intent: dict,
    *,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> list[str]:
    """Build deterministic high-signal query terms for social retrieval."""
    search_filters = intent.get("search_filters") or {}
    terms: list[str] = []

    for value in intent.get("must_match_terms") or []:
        terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    for value in intent.get("entities") or []:
        terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    for value in intent.get("keywords") or []:
        terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    for value in intent.get("domain_terms") or []:
        terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))

    for value in (
        social_filter,
        search_filters.get("social"),
        search_filters.get("tiktok"),
        search_filters.get("instagram"),
    ):
        tokens = re.findall(r"[a-zA-Z0-9_]+", str(value or "").replace("#", " "))
        terms.extend(tokens)

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        token = str(raw or "").strip().lower().replace("_", " ")
        token = re.sub(r"\s+", " ", token)
        token = token.strip()
        if len(token) < 3:
            continue
        if token in _GENERIC_RELEVANCE_TERMS:
            continue
        if token in _PROJECT_TEXT_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)

    if not cleaned:
        context = _project_context_query(project_title, project_description)
        for raw in re.findall(r"[a-zA-Z0-9_]+", context):
            token = str(raw or "").strip().lower().replace("_", " ")
            token = re.sub(r"\s+", " ", token).strip()
            if len(token) < 3:
                continue
            if token in seen:
                continue
            seen.add(token)
            cleaned.append(token)

    return cleaned[:20]


def _prioritize_social_terms(intent: dict, terms: list[str], *, budget: int) -> list[str]:
    """Prioritize must-match/entities first, then fill with remaining terms."""
    budget = max(1, int(budget))
    seen: set[str] = set()
    prioritized: list[str] = []

    priority_terms: list[str] = []
    for value in intent.get("must_match_terms") or []:
        priority_terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    for value in intent.get("entities") or []:
        priority_terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))

    for raw in priority_terms + terms:
        token = str(raw or "").strip().lower().replace("_", " ")
        token = re.sub(r"\s+", " ", token).strip()
        if len(token) < 3:
            continue
        if token in _GENERIC_RELEVANCE_TERMS or token in _PROJECT_TEXT_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        prioritized.append(token)
        if len(prioritized) >= budget:
            break

    return prioritized


def _query_for_social(
    intent: dict,
    *,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Select a robust social search seed even when intent filters are sparse."""
    search_filters = intent.get("search_filters") or {}
    query_budget = max(4, int(settings.INGEST_SOCIAL_QUERY_MAX_TERMS))
    explicit_handles = [
        f"@{h.lstrip('@')}"
        for h in re.findall(
            r"@[A-Za-z0-9_]{2,20}",
            " ".join(
                [
                    str(social_filter or ""),
                    str(search_filters.get("social") or ""),
                    str(project_title or ""),
                    str(project_description or ""),
                ]
            ),
        )
    ]
    explicit_handles = list(dict.fromkeys(explicit_handles))[:2]

    direct = str(
        social_filter or search_filters.get("social") or search_filters.get("instagram") or ""
    ).strip()
    if direct:
        tokens: list[str] = []
        for raw in re.findall(r"[a-zA-Z0-9_]+", direct.replace("#", " ")):
            token = str(raw or "").strip().lower().replace("_", " ")
            token = re.sub(r"\s+", " ", token).strip()
            if len(token) < 3:
                continue
            if token in _GENERIC_RELEVANCE_TERMS or token in _PROJECT_TEXT_STOPWORDS:
                continue
            tokens.append(token)
        merged = list(dict.fromkeys(tokens))
        if len(merged) < 3:
            extras = _social_query_terms(
                intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
            )
            for token in extras:
                if token not in merged:
                    merged.append(token)
        if len(merged) < 4:
            context_terms = re.findall(
                r"[a-zA-Z0-9_]+",
                _project_context_query(project_title, project_description),
            )
            for raw in context_terms:
                token = str(raw or "").strip().lower().replace("_", " ")
                token = re.sub(r"\s+", " ", token).strip()
                if len(token) < 3:
                    continue
                if token in _GENERIC_RELEVANCE_TERMS or token in _PROJECT_TEXT_STOPWORDS:
                    continue
                if token in merged:
                    continue
                merged.append(token)
        merged = _prioritize_social_terms(intent, merged, budget=query_budget)
        if merged:
            return " ".join(explicit_handles + merged).strip()

    terms = _social_query_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    terms = _prioritize_social_terms(intent, terms, budget=query_budget)
    if terms:
        return " ".join(explicit_handles + terms).strip()

    fallback_query = _project_context_query(project_title, project_description)
    return " ".join(explicit_handles + [fallback_query]).strip()


def _query_variants_for_source(intent: dict, base_query: str) -> list[str]:
    """Build bounded query variants: base + focused term/phrase queries from intent."""
    base = str(base_query or "").strip()
    if not base:
        return []
    max_variants = max(1, int(settings.INGEST_QUERY_VARIANTS_PER_SOURCE))
    out: list[str] = []
    seen: set[str] = set()

    def _push(query: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    _push(base)
    focused_phrases: list[str] = []
    for value in (
        *(intent.get("must_match_terms") or []),
        *(intent.get("entities") or []),
        *(intent.get("keywords") or []),
        *(intent.get("domain_terms") or []),
    ):
        phrase = re.sub(r"\s+", " ", str(value or "").replace("#", " ")).strip().lower()
        if not phrase:
            continue
        words = [w for w in re.findall(r"[a-zA-Z0-9_]+", phrase) if len(w) >= 3]
        if not words:
            continue
        normalized_words: list[str] = []
        for word in words:
            lowered = word.lower()
            if lowered in _GENERIC_RELEVANCE_TERMS or lowered in _PROJECT_TEXT_STOPWORDS:
                continue
            normalized_words.append(lowered)
        if not normalized_words:
            continue
        focused = " ".join(normalized_words[:4]).strip()
        if not focused:
            continue
        focused_phrases.append(focused)

    for focused in focused_phrases:
        _push(focused)
        if len(out) >= max_variants:
            break
        _push(f"{focused} {base}")
        if len(out) >= max_variants:
            break

    if len(out) < max_variants:
        token_pool: list[str] = []
        for phrase in focused_phrases:
            token_pool.extend(re.findall(r"[a-zA-Z0-9_]+", phrase))
        for raw in token_pool:
            token = str(raw or "").strip().lower()
            if len(token) < 3:
                continue
            if token in _GENERIC_RELEVANCE_TERMS or token in _PROJECT_TEXT_STOPWORDS:
                continue
            _push(f"{base} {token}")
            if len(out) >= max_variants:
                break
    return out


def _social_query_variants(
    *,
    intent: dict,
    base_query: str,
    social_filter: str,
    project_title: str,
    project_description: str,
    expanded: bool = False,
) -> list[str]:
    """Build bounded social query variants with deterministic term-level fanout."""
    base = str(base_query or "").strip()
    if not base:
        return []

    max_variants = (
        max(1, int(settings.INGEST_SOCIAL_EXPANSION_MAX_VARIANTS))
        if expanded
        else max(1, int(settings.INGEST_SOCIAL_TIER1_MAX_VARIANTS))
    )
    probe_terms = max(0, int(settings.INGEST_SOCIAL_TIER1_PROBE_TERMS))
    out: list[str] = []
    seen: set[str] = set()

    def _push(query: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    _push(base)
    if len(out) >= max_variants:
        return out[:max_variants]

    budget = (
        max(8, int(settings.INGEST_SOCIAL_EXPANDED_QUERY_MAX_TERMS))
        if expanded
        else max(4, int(settings.INGEST_SOCIAL_QUERY_MAX_TERMS))
    )
    terms = _social_query_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    terms = _prioritize_social_terms(intent, terms, budget=budget)
    if not terms:
        for raw in re.findall(r"[a-zA-Z0-9_]+", base):
            token = str(raw or "").strip().lower()
            if len(token) < 3:
                continue
            if token in _GENERIC_RELEVANCE_TERMS or token in _PROJECT_TEXT_STOPWORDS:
                continue
            terms.append(token)
            if len(terms) >= budget:
                break

    for term in terms[:probe_terms]:
        _push(term)
        if len(out) >= max_variants:
            return out[:max_variants]
        _push(f"{base} {term}")
        if len(out) >= max_variants:
            return out[:max_variants]

    if len(terms) >= 2:
        _push(f"{terms[0]} {terms[1]}")
        if len(out) >= max_variants:
            return out[:max_variants]
        _push(f"{base} {terms[0]} {terms[1]}")
        if len(out) >= max_variants:
            return out[:max_variants]

    if len(out) < max_variants:
        for variant in _query_variants_for_source(intent, base):
            _push(variant)
            if len(out) >= max_variants:
                break

    return out[:max_variants]


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
        return direct

    news_fallback = str(search_filters.get("news") or "").strip()
    if news_fallback:
        return news_fallback

    terms = [str(value).strip() for value in entities[:2] if str(value).strip()]
    terms.extend(str(value).strip() for value in keywords[:6] if str(value).strip())
    if terms:
        return " ".join(terms)

    social_fallback = str(social_filter or "").replace("#", " ").strip()
    if social_fallback:
        return social_fallback

    return _project_context_query(project_title, project_description)


def _build_expanded_social_filter(
    intent: dict,
    *,
    social_filter: str,
    project_title: str,
    project_description: str,
) -> str:
    """Build a broader social query used only in low-data expansion pass."""
    budget = max(8, int(settings.INGEST_SOCIAL_EXPANDED_QUERY_MAX_TERMS))
    terms = _social_query_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    terms = _prioritize_social_terms(intent, terms, budget=budget)
    if terms:
        return " ".join(terms)
    return _query_for_social(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )


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
        return direct

    terms: list[str] = []
    if entities:
        terms.append(str(entities[0]))
    terms.extend(str(keyword) for keyword in keywords[:4] if str(keyword).strip())
    if terms:
        return " ".join(term.strip() for term in terms if term.strip())

    return _project_context_query(project_title, project_description)


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
    for query_variant in _query_variants_for_source(intent, query):
        try:
            candidates.extend(await search_exa(f"{query_variant} patents"))
        except Exception:
            logger.exception("ingest_patent_fallback.exa_failed", project_id=project_id)
        try:
            candidates.extend(await search_tavily(f"{query_variant} patents"))
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
        articles: list[dict] = []
        seen_articles: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_perigon(
                query_variant,
                must_match_terms=intent.get("must_match_terms") or [],
                domain_terms=intent.get("domain_terms") or [],
                query_specificity=intent.get("query_specificity"),
            )
            for article in batch:
                article_key = str(
                    article.get("url") or article.get("id") or article.get("title") or ""
                ).strip()
                if not article_key:
                    continue
                normalized_key = article_key.lower()
                if normalized_key in seen_articles:
                    continue
                seen_articles.add(normalized_key)
                articles.append(article)
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
        results: list[dict] = []
        seen_results: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_tavily(query_variant)
            for row in batch:
                result_key = str(row.get("url") or row.get("title") or "").strip()
                if not result_key:
                    continue
                normalized_key = result_key.lower()
                if normalized_key in seen_results:
                    continue
                seen_results.add(normalized_key)
                results.append(row)
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
        results: list[dict] = []
        seen_results: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_exa(query_variant)
            for row in batch:
                result_key = str(row.get("url") or row.get("title") or "").strip()
                if not result_key:
                    continue
                normalized_key = result_key.lower()
                if normalized_key in seen_results:
                    continue
                seen_results.add(normalized_key)
                results.append(row)
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
