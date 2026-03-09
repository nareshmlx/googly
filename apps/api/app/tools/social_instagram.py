"""Instagram tool — EnsembleData SDK wrapper.

Naming convention: {platform}_{resource}_{action}
All functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

EnsembleData observed response times:
  instagram.search       ~1.54s
  instagram.user_posts   ~2.84s
SDK calls are synchronous — wrapped with asyncio.to_thread throughout.
"""

import json
import os
from contextlib import suppress
from typing import Any

import certifi
import httpx
import structlog
from pydantic import BaseModel, Field

from app.core.cache_keys import build_search_cache_key, build_stale_cache_key
from app.core.config import settings
from app.tools.social_common import (
    RobustInt,
    ensemble_sdk_call,
)


class InstagramPostSchema(BaseModel):
    shortcode: str = Field(default="")
    caption: str = Field(default="")
    timestamp: Any = Field(default=None)
    like_count: RobustInt = Field(default=0)
    view_count: RobustInt = Field(default=0)
    cover_url: str = Field(default="")
    video_url: str = Field(default="")
    username: str | None = Field(default=None)

logger = structlog.get_logger(__name__)
_SDK_WARNED = False


def _ensure_ssl_ca_bundle() -> None:
    """Retain the historical local helper name for TLS bootstrap tests/callers."""
    certifi_path = certifi.where()
    if certifi_path and os.path.exists(certifi_path):
        os.environ.setdefault("SSL_CERT_FILE", certifi_path)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi_path)








def _cache_key(project_id: str, tool: str, **kwargs) -> str:
    """Generate a deterministic, project-scoped cache key for Instagram tools."""
    sorted_kwargs = sorted(kwargs.items())
    return build_search_cache_key(
        project_id=project_id,
        provider="social_instagram",
        query_type=tool,
        parts=[f"{k}:{v}" for k, v in sorted_kwargs],
    )


async def instagram_search(project_id: str, text: str, redis=None) -> list[dict]:
    """Search Instagram for users, hashtags, or topics with caching."""
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return []

    cache_key = _cache_key(project_id, "search", text=cleaned_text)
    stale_key = build_stale_cache_key(cache_key)
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
            with suppress(Exception):
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_instagram.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
        logger.warning(
            "social_instagram.search_failed",
            project_id=project_id,
            error=str(exc),
        )
        return []


async def _instagram_search_impl(text: str) -> list[dict]:
    """Internal implementation of Instagram search."""
    result = await ensemble_sdk_call(
        "instagram",
        "instagram.search",
        text=text,
    )
    if not result:
        return []

    data = (result.data or {}) if hasattr(result, "data") else {}
    users_raw = data.get("users", [])
    if not isinstance(users_raw, list):
        return []

    # Each entry wraps the real user object under the "user" key
    return [entry["user"] for entry in users_raw if entry.get("user")]



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
    stale_key = build_stale_cache_key(cache_key)
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
            with suppress(Exception):
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_instagram.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
        logger.warning(
            "social_instagram.user_reels_failed",
            project_id=project_id,
            user_id=user_id,
            error=str(exc),
        )
        return []


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
    try:
        result = await ensemble_sdk_call(
            "instagram",
            "instagram.user_reels",
            user_id=user_id,
            depth=depth,
            include_feed_video=True,
        )
        if not result:
            return []

        reels_raw = (result.data or {}).get("reels", [])
        if not isinstance(reels_raw, list):
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

            model = InstagramPostSchema.model_validate(
                {
                    "shortcode": media.get("code") or str(media.get("pk", "")),
                    "caption": caption,
                    "timestamp": media.get("taken_at"),
                    "like_count": media.get("like_count")
                    or (media.get("edge_media_preview_like") or {}).get("count")
                    or (media.get("edge_liked_by") or {}).get("count") or 0,
                    "view_count": media.get("play_count") or media.get("view_count") or media.get("video_view_count") or 0,
                    "cover_url": cover_url,
                    "video_url": video_url,
                }
            )
            items.append(model.model_dump(exclude_none=True))

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
    stale_key = build_stale_cache_key(cache_key)
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
            with suppress(Exception):
                stale = await redis.get(stale_key)
                if stale:
                    data = json.loads(stale)
                    logger.info(
                        "social_instagram.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return data[0], data[1]
        logger.warning(
            "social_instagram.hashtag_posts_failed",
            project_id=project_id,
            hashtag=hashtag,
            error=str(exc),
        )
        return [], None


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
    _ensure_ssl_ca_bundle()

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
            async with httpx.AsyncClient(timeout=settings.SOCIAL_INSTAGRAM_TIMEOUT_SECONDS) as client:
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

                model = InstagramPostSchema.model_validate(
                    {
                        "shortcode": str(
                            node.get("code") or node.get("shortcode") or node.get("id") or ""
                        ),
                        "caption": _extract_caption_text(node),
                        "timestamp": node.get("taken_at") or node.get("taken_at_timestamp"),
                        "like_count": node.get("like_count")
                        or (node.get("edge_media_preview_like") or {}).get("count")
                        or (node.get("edge_liked_by") or {}).get("count") or 0,
                        "view_count": node.get("play_count") or node.get("view_count") or node.get("video_view_count") or 0,
                        "cover_url": cover_url,
                        "video_url": video_url,
                        "username": username,
                    }
                )
                items.append(model.model_dump(exclude_none=True))

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
