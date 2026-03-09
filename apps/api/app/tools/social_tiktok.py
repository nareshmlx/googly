"""TikTok tool — EnsembleData SDK wrapper.

Naming convention: {platform}_{resource}_{action}
All functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

The EnsembleData SDK is synchronous; all calls are wrapped in asyncio.to_thread
so they do not block the event loop.

EnsembleData observed response times:
  /tiktok/hashtag/search   ~2-5s depending on hashtag volume
Timeout is handled by the underlying SDK; the to_thread call is wrapped in a
try/except so any SDK-level exception is caught and logged gracefully.
"""

import asyncio
import json
import os
from contextlib import suppress
from typing import Literal

import certifi
import httpx
import structlog

from app.core.cache_keys import build_search_cache_key, build_stale_cache_key
from app.core.config import settings
from app.tools.social_common import (
    ensemble_sdk_call,
    ensure_ssl_ca_bundle,
    extract_items_from_payload,
    extract_items_from_sdk_result,
)

logger = structlog.get_logger(__name__)



def _safe_url(url_list: object) -> str:
    """Return the first URL from a url_list, or an empty string if absent."""
    if isinstance(url_list, list) and url_list:
        return str(url_list[0])
    return ""


def _map_video(video: dict) -> dict:
    """
    Map a raw EnsembleData TikTok video dict to the canonical Googly shape.

    Extracts only the fields needed by the KB ingest pipeline.  All nested
    lookups are guarded so a malformed API response never raises — the caller
    receives an empty-string / zero fallback instead.
    """
    author: dict = video.get("author") or {}
    stats: dict = video.get("statistics") or {}
    video_data: dict = video.get("video") or {}
    play_addr: dict = video_data.get("play_addr") or {}
    cover: dict = video_data.get("cover") or {}

    return {
        "video_id": video.get("aweme_id", ""),
        "author_username": author.get("unique_id", ""),
        "description": video.get("desc", ""),
        "likes": stats.get("digg_count", 0),
        "views": stats.get("play_count", 0),
        "cover_url": _safe_url(cover.get("url_list")),
        "video_url": _safe_url(play_addr.get("url_list")),
        "source": "social_tiktok",
    }








def _dedupe_videos(batches: list[list[dict]], *, max_results: int) -> list[dict]:
    """Deduplicate mapped videos by video_id while preserving order."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for batch in batches:
        for video in batch:
            vid_id: str = video.get("video_id") or ""
            if vid_id and vid_id not in seen:
                seen.add(vid_id)
                deduped.append(video)
                if len(deduped) >= max_results:
                    return deduped
    return deduped


def _cache_key(
    project_id: str, hashtags: str, keyword_queries: list[str] | None, max_results: int
) -> str:
    """Generate a deterministic, project-scoped cache key for TikTok searches."""
    return build_search_cache_key(
        project_id=project_id,
        provider="social_tiktok",
        query_type="posts",
        parts=[hashtags, sorted(keyword_queries or []), max_results],
    )


async def fetch_tiktok_posts(
    project_id: str,
    hashtags: str = "",
    *,
    keyword_queries: list[str] | None = None,
    exact_match: bool = False,
    period: Literal["0", "1", "7", "30", "90", "180"] = "30",
    max_results: int = 50,
    redis=None,
) -> list[dict]:
    """Fetch TikTok videos with project-scoped caching."""
    # Check cache first
    cache_key = _cache_key(project_id, hashtags, keyword_queries, max_results)
    stale_key = build_stale_cache_key(cache_key)
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            logger.warning("social_tiktok.cache_read_failed", project_id=project_id)

    try:
        results = await _fetch_tiktok_posts_impl(
            hashtags=hashtags,
            keyword_queries=keyword_queries,
            exact_match=exact_match,
            period=period,
            max_results=max_results,
        )
        if redis and results:
            try:
                await redis.setex(cache_key, settings.CACHE_TTL_SOCIAL, json.dumps(results))
                await redis.setex(stale_key, settings.CACHE_TTL_STALE, json.dumps(results))
            except Exception:
                logger.warning("social_tiktok.cache_write_failed", project_id=project_id)
        return results
    except Exception as exc:
        if redis:
            with suppress(Exception):
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_tiktok.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
        raise


async def _fetch_tiktok_posts_impl(
    hashtags: str = "",
    *,
    keyword_queries: list[str] | None = None,
    exact_match: bool = False,
    period: Literal["0", "1", "7", "30", "90", "180"] = "30",
    max_results: int = 50,
) -> list[dict]:
    """Internal implementation of TikTok video retrieval."""
    """
    Fetch TikTok videos matching the given hashtags via the EnsembleData SDK.

    Parses a space-separated hashtag string (e.g. ``"#PG #beautytrends
    #activeingredients"``), fans out one ``hashtag_search`` call per hashtag,
    deduplicates results by ``video_id``, and returns up to 50 videos.

    The EnsembleData SDK is synchronous; each blocking call is offloaded to a
    thread pool via ``asyncio.to_thread`` so it never stalls the FastAPI event
    loop.  Individual hashtag failures are swallowed and logged — one bad tag
    does not abort the whole batch.

    Returns ``[]`` on any failure, including a missing API token — never raises
    to the caller.

    Args:
        hashtags: Space-separated hashtag string such as
            ``"#PG #beautytrends #activeingredients"``.

    Returns:
        Deduplicated list of up to 50 video dicts on success; empty list on
        any failure.
    """
    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_tiktok.no_api_key")
        return []
    is_sdk_available = sdk_available("tiktok")

    deduped_queries: list[str] = []
    seen_query: set[str] = set()
    for value in keyword_queries or []:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen_query:
            continue
        seen_query.add(key)
        deduped_queries.append(cleaned)

    tags = [t.lstrip("#") for t in hashtags.split() if t.strip("#")]
    logger.info(
        "social_tiktok.fetch.start",
        keyword_query_count=len(deduped_queries),
        tag_count=len(tags),
        exact_match=exact_match,
        period=period,
    )

    batches: list[list[dict]] = []
    if deduped_queries:
        keyword_results = await asyncio.gather(
            *[
                ensemble_sdk_call(
                    "tiktok",
                    "tiktok.keyword_search",
                    keyword=query,
                    period=period,
                    sorting="0",
                    match_exactly=exact_match,
                    get_author_stats=True,
                )
                for query in deduped_queries
            ]
        )
        batches.extend(
            [[_map_video(v) for v in extract_items_from_sdk_result(r)] for r in keyword_results if r]
        )

    deduped = _dedupe_videos(batches, max_results=max_results)
    if deduped_queries and len(deduped) < max_results:
        full_batches = await asyncio.gather(
            *[
                _fetch_keyword_full_http(
                    query,
                    exact_match=exact_match,
                    period=period,
                )
                for query in deduped_queries
            ]
        )
        deduped = _dedupe_videos([deduped, *full_batches], max_results=max_results)

    if len(deduped) < max_results and tags:
        hashtag_results = await asyncio.gather(
            *[ensemble_sdk_call("tiktok", "tiktok.hashtag_search", hashtag=tag) for tag in tags]
        )
        batches_tags = [
            [_map_video(v) for v in extract_items_from_sdk_result(r)] for r in hashtag_results if r
        ]
        deduped = _dedupe_videos([deduped, *batches_tags], max_results=max_results)

    if not deduped and not deduped_queries and not tags:
        logger.warning("social_tiktok.no_queries_parsed", hashtags=hashtags)

    logger.info(
        "social_tiktok.fetch.success",
        keyword_query_count=len(deduped_queries),
        tag_count=len(tags),
        video_count=len(deduped),
    )
    return deduped


async def _fetch_keyword_full_http(
    query: str,
    *,
    exact_match: bool,
    period: Literal["0", "1", "7", "30", "90", "180"],
) -> list[dict]:
    """
    Fallback TikTok keyword retrieval via Ensemble full-search HTTP endpoint.

    Used when SDK keyword search is sparse, to improve recall before strict relevance
    filters are applied in ingest.
    """
    if not settings.ENSEMBLE_API_TOKEN:
        return []
    _ensure_ssl_ca_bundle()

    keyword = str(query or "").strip()
    if not keyword:
        return []

    base_url = settings.ENSEMBLE_API_BASE_URL.rstrip("/")
    request_variants: list[tuple[str, dict[str, str]]] = [
        (
            f"{base_url}/tt/keyword/full-search",
            {
                "token": settings.ENSEMBLE_API_TOKEN,
                "name": keyword,
                "period": period,
                "sorting": "0",
                "match_exactly": str(bool(exact_match)).lower(),
            },
        ),
        (
            f"{base_url}/tiktok/keyword/full-search",
            {
                "token": settings.ENSEMBLE_API_TOKEN,
                "keyword": keyword,
                "period": period,
                "sorting": "0",
                "match_exactly": str(bool(exact_match)).lower(),
                "get_author_stats": "true",
            },
        ),
        (
            f"{base_url}/tiktok/keyword/full_search",
            {
                "token": settings.ENSEMBLE_API_TOKEN,
                "keyword": keyword,
                "period": period,
                "sorting": "0",
                "match_exactly": str(bool(exact_match)).lower(),
                "get_author_stats": "true",
            },
        ),
    ]

    for endpoint, params in request_variants:
        try:
            async with httpx.AsyncClient(
                timeout=float(settings.ENSEMBLE_TIKTOK_FULL_SEARCH_TIMEOUT)
            ) as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                payload = response.json()
            items = extract_items_from_payload(payload)
            if not items:
                continue
            return [_map_video(v) for v in items]
        except Exception as exc:
            logger.warning(
                "social_tiktok.keyword_full_fetch.failed",
                endpoint=endpoint,
                query=keyword[:80],
                exact_match=exact_match,
                error_type=type(exc).__name__,
            )

    return []
