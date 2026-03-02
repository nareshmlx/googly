"""YouTube source ingestion task."""

import asyncio
import math
import re
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
from app.tools.social_youtube import search_youtube_videos

logger = structlog.get_logger(__name__)


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
    redis=None,
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
            search_youtube_videos(project_id, qv, max_results=fetch_limit, redis=redis)
            for qv in variants
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for batch in results:
            if isinstance(batch, Exception):
                logger.exception("ingest_youtube.batch_failed", project_id=project_id)
                continue
            for row in batch:
                source_id = str(
                    row.get("source_id") or row.get("video_id") or row.get("url") or ""
                ).strip()
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
        logger.info(
            "ingest_youtube.relevance_fallback", project_id=project_id, candidate_count=len(videos)
        )
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
                content=content,  # Store full content (chunker handles chunking)
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
