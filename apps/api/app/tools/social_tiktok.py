"""TikTok tool — EnsembleData SDK wrapper.

Naming convention: {platform}_{resource}_{action}
All functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

The EnsembleData SDK is synchronous; all calls are wrapped in asyncio.to_thread
so they do not block the event loop.

EnsembleData observed response times:
  /tiktok/hashtag/search   ~2–5s depending on hashtag volume
Timeout is handled by the underlying SDK; the to_thread call is wrapped in a
try/except so any SDK-level exception is caught and logged gracefully.
"""

import asyncio
from typing import Literal

import httpx
import structlog

try:
    from ensembledata.api import EDClient
except ImportError:  # pragma: no cover - environment-dependent optional dependency
    EDClient = None  # type: ignore[assignment]

from app.core.config import settings

logger = structlog.get_logger(__name__)
_SDK_WARNED = False


def _sdk_available() -> bool:
    """Return whether EnsembleData SDK is importable in this runtime."""
    global _SDK_WARNED
    if EDClient is None:
        if not _SDK_WARNED:
            logger.warning("social_tiktok.sdk_missing")
            _SDK_WARNED = True
        return False
    return True


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


def _extract_items(result) -> list[dict]:
    """Extract list payloads from common EnsembleData response variants."""
    raw_items: list = []
    if result and result.data:
        data = result.data
        if isinstance(data, dict):
            raw_items = data.get("data") or data.get("items") or []
        elif isinstance(data, list):
            raw_items = data
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def _extract_items_from_payload(payload: object) -> list[dict]:
    """Extract list payloads from common HTTP JSON response variants."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, dict):
        raw_items = data.get("data") or data.get("items") or []
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = payload.get("items") or payload.get("results") or []
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def _fetch_hashtag_sync(tag: str) -> list[dict]:
    """
    Call the EnsembleData hashtag_search endpoint synchronously.

    Isolated into its own function so asyncio.to_thread has a clean, picklable
    callable with no closure over async state.

    Args:
        tag: A single hashtag string without the leading ``#``, e.g. ``"beautytrends"``.

    Returns:
        List of mapped video dicts, or ``[]`` on any exception.
    """
    try:
        if not _sdk_available():
            return []
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN or "")
        result = client.tiktok.hashtag_search(hashtag=tag)

        return [_map_video(v) for v in _extract_items(result)]
    except Exception as exc:
        try:
            logger.exception("social_tiktok.hashtag_fetch.failed", tag=tag)
        except Exception:
            logger.error(
                "social_tiktok.hashtag_fetch.failed_fallback",
                tag=str(tag).encode("ascii", "ignore").decode("ascii"),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        return []


def _fetch_keyword_sync(
    query: str,
    *,
    exact_match: bool,
    period: Literal["0", "1", "7", "30", "90", "180"],
) -> list[dict]:
    """Call the EnsembleData keyword_search endpoint synchronously."""
    try:
        if not _sdk_available():
            return []
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN or "")
        result = client.tiktok.keyword_search(
            keyword=query,
            period=period,
            sorting="0",
            match_exactly=exact_match,
            get_author_stats=True,
        )
        return [_map_video(v) for v in _extract_items(result)]
    except Exception as exc:
        try:
            logger.exception("social_tiktok.keyword_fetch.failed", query=query, exact_match=exact_match)
        except Exception:
            logger.error(
                "social_tiktok.keyword_fetch.failed_fallback",
                query=str(query).encode("ascii", "ignore").decode("ascii"),
                exact_match=exact_match,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        return []


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


async def fetch_tiktok_posts(
    hashtags: str = "",
    *,
    keyword_queries: list[str] | None = None,
    exact_match: bool = False,
    period: Literal["0", "1", "7", "30", "90", "180"] = "30",
    max_results: int = 50,
) -> list[dict]:
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
    sdk_available = _sdk_available()

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
    if deduped_queries and sdk_available:
        keyword_batches = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _fetch_keyword_sync,
                    query,
                    exact_match=exact_match,
                    period=period,
                )
                for query in deduped_queries
            ]
        )
        batches.extend(keyword_batches)

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

    if len(deduped) < max_results and tags and sdk_available:
        hashtag_batches = await asyncio.gather(
            *[asyncio.to_thread(_fetch_hashtag_sync, tag) for tag in tags]
        )
        deduped = _dedupe_videos([deduped, *hashtag_batches], max_results=max_results)

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
            items = _extract_items_from_payload(payload)
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
