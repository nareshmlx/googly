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

import structlog
from ensembledata.api import EDClient

from app.core.config import settings

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
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN or "")
        result = client.tiktok.hashtag_search(hashtag=tag)

        raw_items: list = []
        if result.data:
            raw_items = result.data.get("data") or []

        if not isinstance(raw_items, list):
            return []

        return [_map_video(v) for v in raw_items if isinstance(v, dict)]
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


async def fetch_tiktok_posts(hashtags: str) -> list[dict]:
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

    tags = [t.lstrip("#") for t in hashtags.split() if t.strip("#")]
    if not tags:
        logger.warning("social_tiktok.no_tags_parsed", hashtags=hashtags)
        return []

    logger.info("social_tiktok.fetch.start", tags=tags)

    results = await asyncio.gather(*[asyncio.to_thread(_fetch_hashtag_sync, tag) for tag in tags])

    seen: set[str] = set()
    deduped: list[dict] = []
    for batch in results:
        for video in batch:
            vid_id: str = video.get("video_id") or ""
            if vid_id and vid_id not in seen:
                seen.add(vid_id)
                deduped.append(video)
                if len(deduped) >= 50:  # noqa: PLR2004 — max videos per call
                    break
        if len(deduped) >= 50:  # noqa: PLR2004
            break

    logger.info("social_tiktok.fetch.success", tags=tags, video_count=len(deduped))
    return deduped
