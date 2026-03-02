"""TikTok source ingestion task."""

import math
from datetime import UTC, datetime

import structlog

from app.core.config import settings
from app.kb.ingester import RawDocument
from app.tasks.ingest_utils import (
    _as_int,
    _content_quality_score,
    _match_count,
    _project_anchor_terms,
    _query_for_social,
    _required_must_match_count,
    _required_social_match_count,
    _social_must_terms,
    _social_query_terms,
    _social_web_fallback_docs,
)
from app.tools.social_tiktok import fetch_tiktok_posts

logger = structlog.get_logger(__name__)


def _normalize_hashtag_token(raw: str) -> str:
    """Normalize free text into a hashtag-safe token without spaces."""
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    words = cleaned.split()[:3]
    compact = re.sub(r"\s+", "", " ".join(words)).strip().lower()
    return compact if len(compact) >= 3 else ""


async def _ingest_tiktok(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
    max_results_override: int | None = None,
    redis=None,
) -> list[RawDocument]:
    """
    Fetch recent TikTok posts for the handle derived from social_filter.
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
        project_id=project_id,
        hashtags=hashtag_query,
        keyword_queries=keyword_queries,
        exact_match=exact_match,
        period="90" if exact_match else "30",
        max_results=max_results,
        redis=redis,
    )

    if len(posts) < settings.INGEST_TIKTOK_MIN_RELEVANT_RESULTS and exact_match:
        expanded = await fetch_tiktok_posts(
            project_id=project_id,
            hashtags=hashtag_query,
            keyword_queries=keyword_queries,
            exact_match=False,
            period="30",
            max_results=max_results,
            redis=redis,
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

    from app.tasks.ingest_utils import _clean_term_values

    for video in posts:
        description = (video.get("description") or "").strip()
        if not description:
            continue
        if (
            must_terms
            and _match_count(description, set(_clean_term_values(must_terms))) < required_must_hits
        ):
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
                content=description,  # Store full content (chunker handles chunking)
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
