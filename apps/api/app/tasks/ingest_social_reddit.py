"""Reddit source ingestion task."""

import asyncio
import math
from datetime import UTC, datetime

import structlog

from app.core.config import settings
from app.kb.ingester import RawDocument
from app.tasks.ingest_utils import (
    _as_int,
    _build_brand_terms,
    _build_relevance_terms,
    _content_quality_score,
    _match_count,
    _project_anchor_terms,
    _query_for_social,
    _required_social_match_count,
    _social_query_variants,
    _social_web_fallback_docs,
)
from app.tools.social_reddit import search_reddit_posts

logger = structlog.get_logger(__name__)


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
    redis=None,
) -> list[RawDocument]:
    """Fetch Reddit posts and map them into social KB documents."""
    query = _query_for_social(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_reddit.no_query", project_id=project_id)
        return []

    fetch_limit = (
        max(1, int(fetch_limit_override))
        if isinstance(fetch_limit_override, int) and fetch_limit_override > 0
        else settings.INGEST_SOCIAL_FETCH_LIMIT_PER_SOURCE
    )
    keep_limit = max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE)

    posts: list[dict] = []
    seen_ids: set[str] = set()
    try:
        variants = list(
            _social_query_variants(
                intent=intent,
                base_query=query,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
                expanded=expansion_mode,
            )
        )
        tasks = [
            search_reddit_posts(project_id, qv, limit=fetch_limit, redis=redis) for qv in variants
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for batch in results:
            if isinstance(batch, Exception):
                logger.exception("ingest_reddit.batch_failed", project_id=project_id)
                continue
            for row in batch:
                sid = str(row.get("source_id") or row.get("id") or "").strip()
                if not sid or sid in seen_ids:
                    continue
                seen_ids.add(sid)
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
        title = str(post.get("title") or "").strip()
        content = str(post.get("content") or "").strip()
        if not content and not title:
            continue
        merged = f"{title} {content}"
        score = _as_int(post.get("score"))
        comments = _as_int(post.get("comments"))
        relevance_match = int(
            _match_count(merged, anchor_terms)
            + _match_count(merged, relevance_terms)
            + _match_count(merged, brand_terms)
        )
        if relevance_match < min_match:
            continue
        quality_score = _content_quality_score(title, content)
        engagement_score = min(1.0, math.log1p(max(0, score + (comments / 2.0))) / 12.0)

        recency_score = 0.4
        ts = post.get("published_at")
        if ts:
            try:
                normalized = str(ts).replace("Z", "+00:00")
                post_dt = datetime.fromisoformat(normalized)
                if post_dt.tzinfo is None:
                    post_dt = post_dt.replace(tzinfo=UTC)
                days_old = max(0.0, (now_utc - post_dt).total_seconds() / 86400.0)
                recency_score = math.exp(-days_old / 28.0)
            except Exception:
                recency_score = 0.4
        scored.append((relevance_match, quality_score, engagement_score, recency_score, post))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    docs: list[RawDocument] = []
    for _, _, _, _, post in scored:
        sid = str(post.get("source_id") or post.get("id") or "").strip()
        title = str(post.get("title") or "").strip()
        content = str(post.get("content") or "").strip()
        if not sid or (not content and not title):
            continue
        subreddit = str(post.get("subreddit") or "").strip()
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_reddit",
                source_id=sid,
                title=title or "Reddit post",
                content=content
                if content
                else title,  # Store full content (chunker handles chunking)
                metadata={
                    "platform": "reddit",
                    "subreddit": subreddit,
                    "author": post.get("author") or "",
                    "score": _as_int(post.get("score")),
                    "comments": _as_int(post.get("comments")),
                    "url": post.get("url") or "",
                    "published_at": post.get("published_at") or "",
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
