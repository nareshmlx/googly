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


async def instagram_search(text: str) -> list[dict]:
    """
    Search Instagram for users, hashtags, or topics via EnsembleData SDK.

    Used at project creation to discover user accounts matching the project's
    social search filter (e.g., "#veganfashion #sustainablefashion").

    The API returns a "users" list where each entry wraps the user object under
    a nested "user" key: {"position": 0, "user": {"pk": "...", "username": ...}}.
    This function unwraps that nesting and returns the inner user dicts directly.

    Returns a list of user dicts on success.
    Returns [] on any failure — never raises.
    """
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


async def instagram_search_multi(queries: list[str]) -> list[dict]:
    """
    Search Instagram for users across multiple short queries and deduplicate.

    Used as a fallback when the primary instagram search filter returns 0 accounts.
    Each query is searched independently; results are deduplicated by the "pk" field
    so the same account appearing under multiple queries is only returned once.

    Queries should be SHORT — 1–3 words each. Long strings reliably return 0 results
    from the EnsembleData SDK. Pass individual keywords, not combined phrases.

    Returns a combined list of unique user dicts on success.
    Returns [] on any failure — never raises.
    """
    if not queries:
        return []

    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned:
        return []

    results_per_query = await asyncio.gather(
        *(instagram_search(query) for query in cleaned),
        return_exceptions=False,
    )

    seen_pks: set[str] = set()
    combined: list[dict] = []
    for results in results_per_query:
        for user in results:
            pk = str(user.get("pk") or user.get("id") or "")
            if pk and pk not in seen_pks:
                seen_pks.add(pk)
                combined.append(user)

    logger.info(
        "social_instagram.search_multi.done",
        query_count=len(cleaned),
        unique_accounts=len(combined),
    )
    return combined


async def instagram_user_posts(
    user_id: int,
    depth: int = 1,
    oldest_timestamp: int | None = None,
) -> list[dict]:
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
    user_id: int,
    depth: int = 1,
    oldest_timestamp: int | None = None,
) -> list[dict]:
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
                media.get("play_count")
                or media.get("view_count")
                or media.get("video_view_count")
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


async def instagram_user_basic_stats(user_id: int) -> dict:
    """
    Fetch basic public profile stats for an Instagram user.

    Returns a normalised dict:
      - followers (int)
      - following (int)
      - username (str)
      - full_name (str)

    Returns {} on failure.
    """
    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_instagram.no_token")
        return {}
    if not _sdk_available():
        return {}

    logger.info("social_instagram.basic_stats.start", user_id=user_id)
    try:
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN)
        result = await asyncio.to_thread(lambda: client.instagram.user_basic_stats(user_id=user_id))
        data = result.data or {}
        if not isinstance(data, dict):
            logger.warning(
                "social_instagram.basic_stats.unexpected_shape",
                user_id=user_id,
                type=str(type(data)),
            )
            return {}
        stats = {
            "followers": _as_int(data.get("followers")),
            "following": _as_int(data.get("following")),
            "username": str(data.get("username") or ""),
            "full_name": str(data.get("full_name") or ""),
        }
        logger.info(
            "social_instagram.basic_stats.success",
            user_id=user_id,
            followers=stats["followers"],
        )
        return stats
    except Exception:
        logger.exception("social_instagram.basic_stats.unexpected_error", user_id=user_id)
        return {}


async def instagram_post_info(code: str) -> dict:
    """
    Fetch detailed stats for a single Instagram post/reel by shortcode.

    Returns a normalised dict:
      - like_count (int)
      - view_count (int)
      - video_url (str)
      - cover_url (str)

    Returns {} on failure.
    """
    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_instagram.no_token")
        return {}
    if not _sdk_available():
        return {}

    if not code:
        return {}

    logger.info("social_instagram.post_info.start", code=code)
    try:
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN)
        result = await asyncio.to_thread(
            lambda: client.instagram.post_info_and_comments(code=code, num_comments=0)
        )
        data = result.data or {}
        if not isinstance(data, dict):
            logger.warning(
                "social_instagram.post_info.unexpected_shape",
                code=code,
                type=str(type(data)),
            )
            return {}

        like_count = _as_int(
            (data.get("edge_media_preview_like") or {}).get("count")
            or (data.get("edge_liked_by") or {}).get("count")
            or data.get("like_count")
        )
        view_count = _as_int(
            data.get("video_view_count") or data.get("video_play_count") or data.get("view_count")
        )
        details = {
            "like_count": like_count,
            "view_count": view_count,
            "video_url": str(data.get("video_url") or ""),
            "cover_url": str(data.get("thumbnail_src") or data.get("display_url") or ""),
        }
        logger.info(
            "social_instagram.post_info.success",
            code=code,
            like_count=like_count,
            view_count=view_count,
        )
        return details
    except Exception:
        logger.exception("social_instagram.post_info.unexpected_error", code=code)
        return {}


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
        candidate_posts = data.get("posts") or data.get("items") or data.get("data") or []
        if isinstance(candidate_posts, list):
            raw_posts = candidate_posts
        next_cursor = (
            data.get("nextCursor")
            or data.get("next_cursor")
            or data.get("cursor")
            or next_cursor
        )
    elif isinstance(data, list):
        raw_posts = data
    else:
        candidate_posts = payload.get("posts") or payload.get("items") or []
        if isinstance(candidate_posts, list):
            raw_posts = candidate_posts

    return raw_posts, str(next_cursor) if next_cursor else None


async def instagram_hashtag_posts(
    hashtag: str,
    cursor: str | None = None,
    get_author_info: bool = True,
) -> tuple[list[dict], str | None]:
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
                    node.get("play_count")
                    or node.get("view_count")
                    or node.get("video_view_count")
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
