"""YouTube retrieval tool using EnsembleData keyword search endpoint."""

import hashlib
import json
import re

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

YOUTUBE_MAX_RESULTS = 100
YOUTUBE_TIMEOUT_SECONDS = 45.0
ENSEMBLE_BASE_FALLBACK = "https://ensembledata.com/apis"


def _extract_video_id(url: str) -> str:
    """Extract YouTube video id from canonical URL forms."""
    raw = str(url or "")
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", raw)
    if match:
        return match.group(1)
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", raw)
    if match:
        return match.group(1)
    match = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", raw)
    if match:
        return match.group(1)
    match = re.search(r"/embed/([A-Za-z0-9_-]{6,})", raw)
    if match:
        return match.group(1)
    match = re.search(r"/live/([A-Za-z0-9_-]{6,})", raw)
    if match:
        return match.group(1)
    return ""


def _endpoint_candidates() -> list[str]:
    """Return ordered unique candidate endpoints for YouTube keyword search."""
    candidates = [
        f"{settings.ENSEMBLE_API_BASE_URL.rstrip('/')}/youtube/search",
        f"{ENSEMBLE_BASE_FALLBACK.rstrip('/')}/youtube/search",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for endpoint in candidates:
        if endpoint in seen:
            continue
        seen.add(endpoint)
        out.append(endpoint)
    return out


def _extract_items(payload: object) -> list[dict]:
    """Best-effort extraction of list payloads from EnsembleData response envelopes."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    candidates = [
        payload.get("data"),
        payload.get("posts"),
        payload.get("videos"),
        payload.get("results"),
    ]
    for value in candidates:
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for nested_key in ("data", "posts", "videos", "results"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, list):
                    return [row for row in nested_value if isinstance(row, dict)]
    return []


def _text_from_runs(value: object) -> str:
    """Extract text from YouTube renderer fields that use runs/simpleText."""
    if isinstance(value, dict):
        simple = value.get("simpleText")
        if isinstance(simple, str) and simple.strip():
            return simple.strip()
        runs = value.get("runs")
        if isinstance(runs, list):
            parts = []
            for run in runs:
                if isinstance(run, dict):
                    text = str(run.get("text") or "").strip()
                    if text:
                        parts.append(text)
            if parts:
                return "".join(parts).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_row(row: dict) -> dict:
    """Flatten common YouTube result wrappers like videoRenderer or handle raw objects."""
    if "videoRenderer" in row and isinstance(row.get("videoRenderer"), dict):
        vr = row["videoRenderer"]
        thumbnails = (vr.get("thumbnail") or {}).get("thumbnails") or []
        thumbnail_url = ""
        if isinstance(thumbnails, list) and thumbnails:
            # Prefer larger thumbnails if available
            first = thumbnails[-1] if len(thumbnails) > 1 else thumbnails[0]
            if isinstance(first, dict):
                thumbnail_url = str(first.get("url") or "").strip()

        watch_url = (vr.get("navigationEndpoint") or {}).get("commandMetadata", {}).get(
            "webCommandMetadata", {}
        ).get("url") or ""
        if watch_url and str(watch_url).startswith("/"):
            watch_url = f"https://www.youtube.com{watch_url}"

        return {
            "video_id": str(vr.get("videoId") or "").strip(),
            "title": _text_from_runs(vr.get("title")),
            "description": _text_from_runs(vr.get("descriptionSnippet")),
            "channel_name": _text_from_runs(vr.get("longBylineText")),
            "thumbnail_url": thumbnail_url,
            "video_url": str(watch_url).strip(),
            "view_count": _text_from_runs(vr.get("viewCountText")),
            "published": _text_from_runs(vr.get("publishedTimeText")),
            "comment_count": "",
            "like_count": "",
        }

    # Fallback: if row is already an object (sometimes happens with "get_additional_info=True" or different endpoints)
    if "video_id" in row or "videoId" in row:
        return row

    return {}


def _cache_key(project_id: str, query: str, max_results: int) -> str:
    """Generate a deterministic, project-scoped cache key for YouTube searches."""
    query_hash = hashlib.sha256(f"{query}:{max_results}".encode()).hexdigest()[:16]
    return f"search:cache:{project_id}:social_youtube:videos:{query_hash}"


async def search_youtube_videos(
    project_id: str,
    query: str,
    max_results: int = 20,
    redis=None,
) -> list[dict]:
    """Retrieve YouTube videos for a topic with project-scoped caching."""
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return []

    # Check cache first
    cache_key = _cache_key(project_id, cleaned_query, max_results)
    stale_key = f"{cache_key}:stale"
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            logger.warning("social_youtube.cache_read_failed", project_id=project_id)

    try:
        results = await _search_youtube_videos_impl(cleaned_query, max_results)
        if redis and results:
            try:
                await redis.setex(cache_key, settings.CACHE_TTL_SOCIAL, json.dumps(results))
                await redis.setex(stale_key, settings.CACHE_TTL_STALE, json.dumps(results))
            except Exception:
                logger.warning("social_youtube.cache_write_failed", project_id=project_id)
        return results
    except Exception as exc:
        if redis:
            try:
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_youtube.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
            except Exception:
                pass
        raise


async def _search_youtube_videos_impl(
    query: str,
    max_results: int = 20,
) -> list[dict]:
    """Internal implementation of YouTube video retrieval."""
    cleaned_query = query
    if not cleaned_query:
        return []

    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_youtube.no_token")
        return []

    bounded_max = max(1, min(max_results, YOUTUBE_MAX_RESULTS))
    logger.info(
        "social_youtube.search.start",
        query_preview=cleaned_query[:80],
        max_results=bounded_max,
    )

    params: dict[str, str | int | float | bool | None] = {
        "keyword": cleaned_query,
        "depth": int(settings.INGEST_YOUTUBE_SEARCH_DEPTH),
        "period": "week",
        "sorting": "relevance",
        "get_additional_info": False,
        "token": str(settings.ENSEMBLE_API_TOKEN),
    }

    payload: object | None = None
    async with httpx.AsyncClient(timeout=YOUTUBE_TIMEOUT_SECONDS) as client:
        for endpoint in _endpoint_candidates():
            try:
                response = await client.get(
                    endpoint,
                    params=params,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                payload = response.json()
                break
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "social_youtube.endpoint_failed",
                    query_preview=cleaned_query[:80],
                    endpoint=endpoint,
                    status_code=int(exc.response.status_code),
                    response_preview=str(exc.response.text or "")[:200],
                )
                continue
            except Exception as exc:
                logger.warning(
                    "social_youtube.endpoint_failed",
                    query_preview=cleaned_query[:80],
                    endpoint=endpoint,
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )
                continue
    if payload is None:
        logger.warning("social_youtube.search.failed", query_preview=cleaned_query[:80])
        return []

    rows = _extract_items(payload)
    results: list[dict] = []
    seen_ids: set[str] = set()
    for row in rows:
        normalized = _normalize_row(row)
        if not normalized:
            continue

        video_url = str(
            normalized.get("video_url")
            or normalized.get("url")
            or normalized.get("watch_url")
            or normalized.get("webUrl")
            or ""
        ).strip()
        source_id = str(
            normalized.get("video_id")
            or normalized.get("videoId")
            or normalized.get("youtube_id")
            or normalized.get("id")
            or _extract_video_id(video_url)
        ).strip()
        if not source_id or source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        title = str(normalized.get("title") or normalized.get("description") or "").strip()
        content = str(normalized.get("description") or normalized.get("title") or "").strip()
        if not content:
            continue
        results.append(
            {
                "source_id": source_id,
                "title": title or "YouTube video",
                "content": content,
                "author": str(
                    normalized.get("channel_name")
                    or normalized.get("author")
                    or normalized.get("owner_username")
                    or ""
                ).strip(),
                "thumbnail_url": str(
                    normalized.get("thumbnail_url")
                    or normalized.get("cover_url")
                    or normalized.get("image_url")
                    or ""
                ).strip(),
                "url": video_url,
                "views": normalized.get("view_count") or normalized.get("views") or 0,
                "likes": normalized.get("like_count") or normalized.get("likes") or 0,
                "comments": normalized.get("comment_count") or normalized.get("comments") or 0,
                "published_at": normalized.get("published")
                or normalized.get("published_at")
                or normalized.get("create_time")
                or "",
            }
        )
        if len(results) >= bounded_max:
            break

    logger.info(
        "social_youtube.search.success", query_preview=cleaned_query[:80], count=len(results)
    )
    return results
