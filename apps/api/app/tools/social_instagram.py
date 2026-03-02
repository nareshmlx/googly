"""Instagram tool — EnsembleData SDK wrapper.

Naming convention: {platform}_{resource}_{action}
All functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

EnsembleData observed response times:
  instagram.search       ~1.54s
  instagram.user_posts   ~2.84s
SDK calls are synchronous — wrapped with asyncio.to_thread throughout.
"""

import asyncio
import hashlib
import json

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
            logger.warning("social_instagram.sdk_missing")
            _SDK_WARNED = True
        return False
    return True


def _as_int(value: object) -> int:
    """Coerce SDK numeric fields that can arrive as int/float/string into int."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        trimmed = value.strip().replace(",", "")
        if not trimmed:
            return 0
        try:
            return int(float(trimmed))
        except ValueError:
            return 0
    return 0


def _cache_key(project_id: str, tool: str, **kwargs) -> str:
    """Generate a deterministic, project-scoped cache key for Instagram tools."""
    sorted_kwargs = sorted(kwargs.items())
    kw_str = ",".join(f"{k}:{v}" for k, v in sorted_kwargs)
    query_hash = hashlib.sha256(kw_str.encode()).hexdigest()[:16]
    return f"search:cache:{project_id}:social_instagram:{tool}:{query_hash}"


async def instagram_search(project_id: str, text: str, redis=None) -> list[dict]:
    """Search Instagram for users, hashtags, or topics with caching."""
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return []

    cache_key = _cache_key(project_id, "search", text=cleaned_text)
    stale_key = f"{cache_key}:stale"
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            logger.warning("social_instagram.cache_read_failed", project_id=project_id)

    try:
        results = await _instagram_search_impl(cleaned_text)
        if redis and results:
            try:
                await redis.setex(cache_key, settings.CACHE_TTL_SOCIAL, json.dumps(results))
                await redis.setex(stale_key, settings.CACHE_TTL_STALE, json.dumps(results))
            except Exception:
                logger.warning("social_instagram.cache_write_failed", project_id=project_id)
        return results
    except Exception as exc:
        if redis:
            try:
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_instagram.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
            except Exception:
                pass
        raise


async def _instagram_search_impl(text: str) -> list[dict]:
    """Internal implementation of Instagram search."""

    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_instagram.no_token")
        return []
    if not _sdk_available():
        return []

    logger.info("social_instagram.search.start", text_preview=text[:60])
    try:
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN)
        result = await asyncio.to_thread(lambda: client.instagram.search(text=text))
        users_raw = (result.data or {}).get("users", [])
        if not isinstance(users_raw, list):
            logger.warning(
                "social_instagram.search.unexpected_shape",
                keys=list((result.data or {}).keys()),
            )
            return []
        # Each entry wraps the real user object under the "user" key
        items = [entry["user"] for entry in users_raw if entry.get("user")]
        logger.info("social_instagram.search.success", result_count=len(items))
        return items
    except Exception:
        safe_preview = text.encode("ascii", "ignore").decode("ascii")[:60]
        try:
            logger.exception("social_instagram.search.unexpected_error", text_preview=safe_preview)
        except Exception:
            logger.error("social_instagram.search.unexpected_error_fallback")
        return []


async def instagram_user_posts(
    project_id: str,
    user_id: int,
    depth: int = 1,
    oldest_timestamp: int | None = None,
    redis=None,
) -> list[dict]:
    """Fetch posts for an Instagram user with caching."""
    cache_key = _cache_key(
        project_id, "user_posts", user_id=user_id, depth=depth, oldest=oldest_timestamp
    )
    stale_key = f"{cache_key}:stale"
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            logger.warning("social_instagram.cache_read_failed", project_id=project_id)

    try:
        results = await _instagram_user_posts_impl(user_id, depth, oldest_timestamp)
        if redis and results:
            try:
                await redis.setex(cache_key, settings.CACHE_TTL_SOCIAL, json.dumps(results))
                await redis.setex(stale_key, settings.CACHE_TTL_STALE, json.dumps(results))
            except Exception:
                logger.warning("social_instagram.cache_write_failed", project_id=project_id)
        return results
    except Exception as exc:
        if redis:
            try:
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_instagram.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
            except Exception:
                pass
        raise


async def _instagram_user_posts_impl(
    user_id: int,
    depth: int = 1,
    oldest_timestamp: int | None = None,
) -> list[dict]:
    """Internal implementation of Instagram user posts retrieval."""

    """
    Fetch posts for an Instagram user via EnsembleData SDK instagram.user_posts.

    depth controls how many pages to scroll (1 = first batch, ~10-12 items).
    oldest_timestamp stops fetching when a post older than that Unix timestamp
    is encountered — pass int(last_refreshed_at.timestamp()) on refresh runs
    to avoid re-ingesting content already in the KB.

    Returns list of post dicts on success. Returns [] on any failure — never raises.
    """
    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_instagram.no_token")
        return []
    if not _sdk_available():
        return []

    logger.info("social_instagram.posts.start", user_id=user_id, depth=depth)
    try:
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN)
        result = await asyncio.to_thread(
            lambda: client.instagram.user_posts(user_id=user_id, depth=depth)
        )
        # EnsembleData response shape: {"count": N, "posts": [{"node": {...}}, ...], "last_cursor": "..."}
        posts_raw = (result.data or {}).get("posts", [])
        if not isinstance(posts_raw, list):
            logger.warning(
                "social_instagram.posts.unexpected_shape",
                keys=list((result.data or {}).keys()),
            )
            return []

        # Unwrap the GraphQL node envelope and normalise field names
        items = []
        for entry in posts_raw:
            node = entry.get("node") if isinstance(entry, dict) else None
            if not isinstance(node, dict):
                continue
            caption_edges = (node.get("edge_media_to_caption") or {}).get("edges", [])
            caption = (
                caption_edges[0].get("node", {}).get("text", "")
                if caption_edges
                else (node.get("accessibility_caption") or "")
            )
            items.append(
                {
                    "shortcode": node.get("shortcode") or node.get("id", ""),
                    "caption": caption,
                    "timestamp": node.get("taken_at_timestamp"),
                    "like_count": (node.get("edge_media_preview_like") or {}).get("count", 0),
                    "view_count": node.get("video_view_count"),
                    "display_url": node.get("display_url", ""),
                    "video_url": node.get("video_url", ""),
                }
            )

        # Final-filter: remove items older than oldest_timestamp.
        # The API may include the stop-trigger post itself (per EnsembleData docs tip).
        if oldest_timestamp is not None:
            items = [
                item
                for item in items
                if isinstance(item.get("timestamp"), int | float)
                and item["timestamp"] >= oldest_timestamp
            ]

        logger.info("social_instagram.posts.success", post_count=len(items), user_id=user_id)
        return items
    except Exception:
        logger.exception("social_instagram.posts.unexpected_error", user_id=user_id)
        return []


async def instagram_user_reels(
    project_id: str,
    user_id: int,
    depth: int = 1,
    oldest_timestamp: int | None = None,
    redis=None,
) -> list[dict]:
    """Fetch video reels for an Instagram user with caching."""
    cache_key = _cache_key(
        project_id, "user_reels", user_id=user_id, depth=depth, oldest=oldest_timestamp
    )
    stale_key = f"{cache_key}:stale"
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            logger.warning("social_instagram.cache_read_failed", project_id=project_id)

    try:
        results = await _instagram_user_reels_impl(user_id, depth, oldest_timestamp)
        if redis and results:
            try:
                await redis.setex(cache_key, settings.CACHE_TTL_SOCIAL, json.dumps(results))
                await redis.setex(stale_key, settings.CACHE_TTL_STALE, json.dumps(results))
            except Exception:
                logger.warning("social_instagram.cache_write_failed", project_id=project_id)
        return results
    except Exception as exc:
        if redis:
            try:
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_instagram.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
            except Exception:
                pass
        raise


async def _instagram_user_reels_impl(
    user_id: int,
    depth: int = 1,
    oldest_timestamp: int | None = None,
) -> list[dict]:
    """Internal implementation of Instagram user reels retrieval."""

    """
    Fetch video reels for an Instagram user via EnsembleData SDK instagram.user_reels.

    Unlike user_posts (which returns static images), user_reels returns actual video
    content with a video_url — enabling inline playback on the Discover page.

    depth controls how many pages to scroll (1 = first batch, ~12 reels).
    oldest_timestamp stops fetching when a reel older than that Unix timestamp
    is encountered — used on refresh runs to avoid re-ingesting known content.

    Returns list of normalised reel dicts on success. Returns [] on any failure — never raises.
    """
    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_instagram.no_token")
        return []
    if not _sdk_available():
        return []

    logger.info("social_instagram.reels.start", user_id=user_id, depth=depth)
    try:
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN)
        result = await asyncio.to_thread(
            lambda: client.instagram.user_reels(
                user_id=user_id,
                depth=depth,
                include_feed_video=True,
            )
        )
        reels_raw = (result.data or {}).get("reels", [])
        if not isinstance(reels_raw, list):
            logger.warning(
                "social_instagram.reels.unexpected_shape",
                keys=list((result.data or {}).keys()),
            )
            return []

        items = []
        for entry in reels_raw:
            media = entry.get("media") if isinstance(entry, dict) else None
            if not isinstance(media, dict):
                continue

            # Extract caption text
            caption_obj = media.get("caption") or {}
            caption = caption_obj.get("text", "") if isinstance(caption_obj, dict) else ""

            # Extract cover URL (first candidate from image_versions2)
            image_candidates = (media.get("image_versions2") or {}).get("candidates", [])
            cover_url = image_candidates[0].get("url", "") if image_candidates else ""

            # Extract video URL (first version from video_versions)
            video_versions = media.get("video_versions") or []
            video_url = video_versions[0].get("url", "") if video_versions else ""

            like_count = _as_int(
                media.get("like_count")
                or (media.get("edge_media_preview_like") or {}).get("count")
                or (media.get("edge_liked_by") or {}).get("count")
            )
            view_count = _as_int(
                media.get("play_count") or media.get("view_count") or media.get("video_view_count")
            )

            items.append(
                {
                    "shortcode": media.get("code") or str(media.get("pk", "")),
                    "caption": caption,
                    "timestamp": media.get("taken_at"),
                    "like_count": like_count,
                    "view_count": view_count,
                    "cover_url": cover_url,
                    "video_url": video_url,
                }
            )

        # Filter out reels older than oldest_timestamp
        if oldest_timestamp is not None:
            filtered_items: list[dict] = []
            for item in items:
                ts = item.get("timestamp")
                if isinstance(ts, int | float) and ts >= oldest_timestamp:
                    filtered_items.append(item)
            items = filtered_items

        logger.info("social_instagram.reels.success", reel_count=len(items), user_id=user_id)
        return items
    except Exception:
        logger.exception("social_instagram.reels.unexpected_error", user_id=user_id)
        return []


def _extract_caption_text(node: dict) -> str:
    """Extract caption text from varying Instagram payload shapes."""
    caption_obj = node.get("caption")
    if isinstance(caption_obj, dict):
        text = caption_obj.get("text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(caption_obj, str):
        return caption_obj.strip()
    caption_edges = (node.get("edge_media_to_caption") or {}).get("edges", [])
    if caption_edges:
        text = (caption_edges[0].get("node") or {}).get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _extract_hashtag_posts_payload(payload: object) -> tuple[list[dict], str | None]:
    """Normalize hashtag posts payload variants returned by Ensemble endpoints."""
    if not isinstance(payload, dict):
        return [], None

    data = payload.get("data")
    raw_posts: list[dict] = []
    next_cursor = payload.get("nextCursor") or payload.get("next_cursor") or payload.get("cursor")

    if isinstance(data, dict):
        recent_posts = data.get("recent_posts")
        top_posts = data.get("top_posts")
        candidate_posts = (
            data.get("posts")
            or data.get("items")
            or data.get("data")
            or recent_posts
            or top_posts
            or []
        )
        if (isinstance(recent_posts, list) or isinstance(top_posts, list)) and not isinstance(
            candidate_posts, list
        ):
            candidate_posts = []
        if isinstance(recent_posts, list) or isinstance(top_posts, list):
            merged_posts: list[dict] = []
            if isinstance(top_posts, list):
                merged_posts.extend([p for p in top_posts if isinstance(p, dict)])
            if isinstance(recent_posts, list):
                merged_posts.extend([p for p in recent_posts if isinstance(p, dict)])
            if merged_posts:
                candidate_posts = merged_posts
        if isinstance(candidate_posts, list):
            raw_posts = candidate_posts
        next_cursor = (
            data.get("nextCursor") or data.get("next_cursor") or data.get("cursor") or next_cursor
        )
    elif isinstance(data, list):
        raw_posts = data
    else:
        candidate_posts = payload.get("posts") or payload.get("items") or []
        if isinstance(candidate_posts, list):
            raw_posts = candidate_posts

    return raw_posts, str(next_cursor) if next_cursor else None


async def instagram_hashtag_posts(
    project_id: str,
    hashtag: str,
    cursor: str | None = None,
    get_author_info: bool = True,
    redis=None,
) -> tuple[list[dict], str | None]:
    """Fetch Instagram posts for a hashtag with caching."""
    cache_key = _cache_key(
        project_id, "hashtag_posts", hashtag=hashtag, cursor=cursor, author=get_author_info
    )
    stale_key = f"{cache_key}:stale"
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                # Cache stores a tuple as list [posts, cursor]
                data = json.loads(cached)
                return data[0], data[1]
        except Exception:
            logger.warning("social_instagram.cache_read_failed", project_id=project_id)

    try:
        items, next_cursor = await _instagram_hashtag_posts_impl(hashtag, cursor, get_author_info)
        if redis and items:
            try:
                await redis.setex(
                    cache_key, settings.CACHE_TTL_SOCIAL, json.dumps([items, next_cursor])
                )
                await redis.setex(
                    stale_key, settings.CACHE_TTL_STALE, json.dumps([items, next_cursor])
                )
            except Exception:
                logger.warning("social_instagram.cache_write_failed", project_id=project_id)
        return items, next_cursor
    except Exception as exc:
        if redis:
            try:
                stale = await redis.get(stale_key)
                if stale:
                    data = json.loads(stale)
                    logger.info(
                        "social_instagram.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return data[0], data[1]
            except Exception:
                pass
        raise


async def _instagram_hashtag_posts_impl(
    hashtag: str,
    cursor: str | None = None,
    get_author_info: bool = True,
) -> tuple[list[dict], str | None]:
    """Internal implementation of Instagram hashtag posts retrieval."""
    """
    Fetch Instagram posts for a hashtag using EnsembleData hashtag endpoint.

    This endpoint is not exposed in the current Python SDK wrapper, so it is
    called directly over HTTP.

    Returns:
      - (items, next_cursor) on success
      - ([], None) on any failure
    """
    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_instagram.no_token")
        return [], None

    tag = hashtag.strip().lstrip("#")
    if not tag:
        return [], None

    base_url = settings.ENSEMBLE_API_BASE_URL.rstrip("/")
    endpoint_candidates = [
        f"{base_url}/instagram/hashtag/posts",
        "https://ensembledata.com/apis/instagram/hashtag/posts",
        "https://api.ensembledata.com/instagram/hashtag/posts",
    ]
    seen_endpoints: set[str] = set()
    endpoints: list[str] = []
    for endpoint in endpoint_candidates:
        if endpoint in seen_endpoints:
            continue
        seen_endpoints.add(endpoint)
        endpoints.append(endpoint)

    params: dict[str, str] = {
        "token": settings.ENSEMBLE_API_TOKEN,
        "name": tag,
        "get_author_info": str(get_author_info).lower(),
    }
    if cursor:
        params["cursor"] = cursor

    logger.info("social_instagram.hashtag_posts.start", hashtag=tag, has_cursor=bool(cursor))

    for endpoint in endpoints:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                payload = response.json()

            raw_posts, next_cursor = _extract_hashtag_posts_payload(payload)
            if not raw_posts and next_cursor is not None:
                logger.info(
                    "social_instagram.hashtag_posts.empty_page_with_cursor",
                    hashtag=tag,
                    endpoint=endpoint,
                )
                continue
            if not raw_posts and next_cursor is None:
                logger.warning(
                    "social_instagram.hashtag_posts.unexpected_shape",
                    hashtag=tag,
                    endpoint=endpoint,
                    payload_type=str(type(payload)),
                )
                continue

            items: list[dict] = []
            for entry in raw_posts:
                if not isinstance(entry, dict):
                    continue
                node = entry.get("node") if isinstance(entry.get("node"), dict) else entry
                if not isinstance(node, dict):
                    continue

                user_obj = node.get("user") or node.get("owner") or {}
                username = ""
                if isinstance(user_obj, dict):
                    username = str(user_obj.get("username") or user_obj.get("handle") or "").strip()
                if not username:
                    username = str(entry.get("username") or "").strip()

                image_candidates = (node.get("image_versions2") or {}).get("candidates", [])
                cover_url = ""
                if isinstance(image_candidates, list) and image_candidates:
                    first = image_candidates[0]
                    if isinstance(first, dict):
                        cover_url = str(first.get("url") or "")
                if not cover_url:
                    cover_url = str(node.get("display_url") or node.get("thumbnail_src") or "")

                video_versions = node.get("video_versions") or []
                video_url = ""
                if isinstance(video_versions, list) and video_versions:
                    first_video = video_versions[0]
                    if isinstance(first_video, dict):
                        video_url = str(first_video.get("url") or "")
                if not video_url:
                    video_url = str(node.get("video_url") or "")

                like_count = _as_int(
                    node.get("like_count")
                    or (node.get("edge_media_preview_like") or {}).get("count")
                    or (node.get("edge_liked_by") or {}).get("count")
                )
                view_count = _as_int(
                    node.get("play_count") or node.get("view_count") or node.get("video_view_count")
                )

                items.append(
                    {
                        "shortcode": str(
                            node.get("code") or node.get("shortcode") or node.get("id") or ""
                        ),
                        "caption": _extract_caption_text(node),
                        "timestamp": node.get("taken_at") or node.get("taken_at_timestamp"),
                        "like_count": like_count,
                        "view_count": view_count,
                        "cover_url": cover_url,
                        "video_url": video_url,
                        "username": username,
                    }
                )

            logger.info(
                "social_instagram.hashtag_posts.success",
                hashtag=tag,
                endpoint=endpoint,
                post_count=len(items),
                has_next_cursor=bool(next_cursor),
            )
            return items, next_cursor
        except Exception as exc:
            logger.warning(
                "social_instagram.hashtag_posts.endpoint_failed",
                hashtag=tag,
                endpoint=endpoint,
                error_type=type(exc).__name__,
            )
            continue

    logger.error("social_instagram.hashtag_posts.all_endpoints_failed", hashtag=tag)
    return [], None
