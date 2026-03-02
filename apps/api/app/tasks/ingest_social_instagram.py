"""Instagram source ingestion task."""

import asyncio
import math
from datetime import UTC, datetime, timedelta

import structlog

from app.core.config import settings
from app.kb.ingester import RawDocument
from app.tasks.ingest_utils import (
    REEL_RECENCY_DAYS,
    _as_int,
    _build_brand_terms,
    _build_relevance_terms,
    _content_quality_score,
    _match_count,
    _project_anchor_terms,
    _query_for_social,
    _social_web_fallback_docs,
)
from app.tools.social_instagram import (
    instagram_hashtag_posts,
    instagram_search,
    instagram_user_reels,
)

logger = structlog.get_logger(__name__)


def _clean_instagram_keyword_query(raw: str) -> str:
    """
    Convert a raw keyword into a safe short Instagram search query.
    """
    import re

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
    import re

    compact = re.sub(r"\s+", "", cleaned).strip().lower()
    return compact if len(compact) >= 3 else ""


def _extract_hashtags(text: str) -> list[str]:
    """Extract cleaned hashtag terms from a social filter string."""
    if not text:
        return []
    import re

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


def _instagram_handle_candidates(social_filter: str, intent: dict) -> list[str]:
    """Build candidate Instagram handle queries for fallback account lookup."""
    import re

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


async def _fetch_single_candidate(
    query: str, project_id: str, oldest_timestamp: int | None, redis
) -> list[dict]:
    try:
        search_results = await instagram_search(project_id, query, redis=redis)
    except Exception:
        logger.exception(
            "ingest_instagram.fallback_search_failed", project_id=project_id, query=query
        )
        return []
    if not search_results:
        return []

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
        return []

    try:
        uid = int(str(raw_uid))
    except (TypeError, ValueError):
        return []

    username = str(candidate.get("username") or query).strip()
    try:
        reels = await instagram_user_reels(
            project_id, uid, depth=1, oldest_timestamp=oldest_timestamp, redis=redis
        )
    except Exception:
        logger.exception(
            "ingest_instagram.fallback_reels_failed", project_id=project_id, query=query, uid=uid
        )
        return []

    reels_out = []
    for reel in reels or []:
        if not reel.get("username"):
            reel["username"] = username
        reels_out.append(reel)
    return reels_out


async def _fetch_hashtag_pages(
    hashtag: str, max_pages: int, project_id: str, recency_cutoff: datetime, redis
) -> list[dict]:
    all_posts = []
    cursor = None
    empty_page_streak = 0
    for _ in range(max_pages):
        try:
            posts, next_cursor = await instagram_hashtag_posts(
                project_id=project_id,
                hashtag=hashtag.replace(" ", ""),
                cursor=cursor,
                get_author_info=True,
                redis=redis,
            )
        except Exception:
            logger.exception(
                "ingest_instagram.hashtag_failed", project_id=project_id, hashtag=hashtag
            )
            break

        if not posts and next_cursor:
            empty_page_streak += 1
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

        if next_cursor == cursor or not next_cursor:
            break
        cursor = next_cursor
    return all_posts


async def _fetch_instagram_reels_from_candidates(
    *,
    project_id: str,
    candidate_queries: list[str],
    oldest_timestamp: int | None,
    redis=None,
) -> list[dict]:
    """Fetch reels by searching candidate handles and resolving account IDs."""
    reels_out: list[dict] = []
    seen_users: set[int] = set()

    tasks = [
        _fetch_single_candidate(query, project_id, oldest_timestamp, redis)
        for query in candidate_queries
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for batch in results:
        if isinstance(batch, Exception):
            continue
        for reel in batch:
            uid = reel.get("user", {}).get("pk") or reel.get("owner", {}).get("id")
            if uid:
                try:
                    uid = int(str(uid))
                    if uid in seen_users:
                        continue
                    seen_users.add(uid)
                except (TypeError, ValueError):
                    pass
            reels_out.append(reel)
            if len(reels_out) >= max(20, settings.INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT):
                break
        if len(reels_out) >= max(20, settings.INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT):
            break

    logger.info(
        "ingest_instagram.fallback_done",
        project_id=project_id,
        candidate_queries=len(candidate_queries),
        reel_count=len(reels_out),
    )
    return reels_out


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
            redis=redis,
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
        tasks = [
            _fetch_hashtag_pages(hashtag, max_pages, project_id, recency_cutoff, redis)
            for hashtag in hashtags[: max(1, settings.INGEST_INSTAGRAM_HASHTAG_QUERIES)]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for batch in results:
            if isinstance(batch, Exception):
                logger.exception("ingest_instagram.hashtag_gather_failed", project_id=project_id)
                continue
            all_posts.extend(batch)

        # OPTIMIZATION: Filter for videos only if enabled
        # Hashtag posts include video_url for video content - this saves expensive user lookup
        if settings.INGEST_INSTAGRAM_VIDEO_ONLY and all_posts:
            video_posts = [post for post in all_posts if post.get("video_url")]
            logger.info(
                "ingest_instagram.video_filter",
                project_id=project_id,
                total_posts=len(all_posts),
                video_posts=len(video_posts),
            )
            if video_posts:
                all_posts = video_posts

    if not all_posts:
        fallback_reels = await _fetch_instagram_reels_from_candidates(
            project_id=project_id,
            candidate_queries=handle_candidates,
            oldest_timestamp=oldest_timestamp,
            redis=redis,
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

    # OPTIMIZATION: Skip expensive user lookup + reels if using hashtag-only mode
    # Hashtag posts with video filter already give us video content from many accounts
    if settings.INGEST_INSTAGRAM_USE_HASHTAG_ONLY:
        logger.info(
            "ingest_instagram.hashtag_only_mode",
            project_id=project_id,
            posts_count=len(all_posts),
        )
        all_reels = list(all_posts)
    else:
        # Legacy mode: extract top authors and fetch their reels (expensive!)
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

        all_reels: list[dict] = list(all_posts)
        tasks = [
            _fetch_single_candidate(username, project_id, oldest_timestamp, redis)
            for username in top_authors
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for batch in results:
            if isinstance(batch, Exception):
                logger.exception("ingest_instagram.top_author_gather_failed", project_id=project_id)
                continue
            all_reels.extend(batch)

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
        engagement_score = (
            min(1.0, math.log1p(max(0, likes + views)) / 20.0) if likes + views > 0 else 0.0
        )

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
        min(
            settings.INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT,
            settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE,
        ),
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
                content=caption,  # Store full content (chunker handles chunking)
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

    logger.info("ingest_instagram.success", project_id=project_id, count=len(docs))
    if not docs:
        return await _social_web_fallback_docs(
            source="social_instagram",
            project_id=project_id,
            user_id=user_id,
            query=fallback_query,
            keep_limit=max(1, settings.INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE),
        )
    return docs
