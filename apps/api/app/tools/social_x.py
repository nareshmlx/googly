"""X (Twitter) retrieval tool backed by EnsembleData endpoints."""

import base64
import hashlib
import json
import re

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

X_MAX_RESULTS = 100
X_TIMEOUT_SECONDS = 20.0
ENSEMBLE_BASE_FALLBACK = "https://ensembledata.com/apis"
X_HANDLE_STOPWORDS = frozenset(
    {
        "twitter",
        "reddit",
        "youtube",
        "social",
        "latest",
        "trend",
        "trends",
        "news",
        "beauty",
        "fashion",
        "skincare",
    }
)


class _XAuthRequiredError(RuntimeError):
    """Raised when X keyword endpoints require unavailable auth/session fields."""


def _is_auth_required_error(status_code: int | None, payload_or_text: object) -> bool:
    """Detect auth-required responses from Ensemble keyword endpoints."""
    text = str(payload_or_text or "").lower()
    if status_code in {401, 403}:
        return True
    if status_code == 422 and ("ct0" in text or "auth_token" in text or "guest_id" in text):
        return True
    return "authentication required" in text or "authorization required" in text


def _keyword_auth_params() -> dict[str, str]:
    """Return optional Ensemble X keyword auth/session params when configured."""
    params: dict[str, str] = {}
    if settings.ENSEMBLE_X_CT0:
        params["ct0"] = settings.ENSEMBLE_X_CT0
    if settings.ENSEMBLE_X_AUTH_TOKEN:
        params["auth_token"] = settings.ENSEMBLE_X_AUTH_TOKEN
    if settings.ENSEMBLE_X_GUEST_ID:
        params["guest_id"] = settings.ENSEMBLE_X_GUEST_ID
    return params


def _as_int(value: object) -> int:
    """Convert mixed metric values (int/float/str) into int safely."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return 0
        try:
            return int(float(cleaned))
        except ValueError:
            return 0
    return 0


def _extract_items(payload: object) -> list[dict]:
    """Extract list records from common API envelope shapes."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    candidates = [
        payload.get("data"),
        payload.get("posts"),
        payload.get("tweets"),
        payload.get("results"),
    ]
    for value in candidates:
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for nested_key in ("data", "posts", "tweets", "results"):
                nested = value.get(nested_key)
                if isinstance(nested, list):
                    return [row for row in nested if isinstance(row, dict)]
    return []


def _normalize_user_id(value: object) -> str:
    """Convert user id variants to numeric id expected by /twitter/user/tweets."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return raw
    match = re.search(r"(\d{3,})", raw)
    if match:
        return match.group(1)
    try:
        decoded = base64.b64decode(raw).decode("utf-8", errors="ignore")
        match = re.search(r"(\d{3,})", decoded)
        if match:
            return match.group(1)
    except Exception:
        pass
    return ""


def _dig(payload: object, path: tuple[str, ...]) -> object | None:
    """Safely fetch nested dict path."""
    cur = payload
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _normalize_tweet_row(post: dict) -> dict:
    """Normalize Ensemble twitter timeline row into a flat tweet-like dict."""
    # Flat shapes are passed through.
    if post.get("text") or post.get("full_text"):
        return post

    result = _dig(
        post,
        (
            "content",
            "itemContent",
            "tweet_results",
            "result",
        ),
    )
    if not isinstance(result, dict):
        return post

    legacy_obj = result.get("legacy")
    legacy: dict = legacy_obj if isinstance(legacy_obj, dict) else {}
    core_user = _dig(result, ("core", "user_results", "result"))
    user_legacy_obj = core_user.get("legacy") if isinstance(core_user, dict) else None
    user_legacy: dict = user_legacy_obj if isinstance(user_legacy_obj, dict) else {}

    screen_name = str(user_legacy.get("screen_name") or "").strip()
    tweet_id = str(result.get("rest_id") or legacy.get("id_str") or "").strip()
    text = str(legacy.get("full_text") or legacy.get("text") or "").strip()
    if not text:
        text = str(
            _dig(result, ("note_tweet", "note_tweet_results", "result", "text")) or ""
        ).strip()
    url = f"https://x.com/{screen_name}/status/{tweet_id}" if screen_name and tweet_id else ""

    return {
        "id": tweet_id,
        "text": text,
        "screen_name": screen_name,
        "name": str(user_legacy.get("name") or "").strip(),
        "favorite_count": legacy.get("favorite_count") or 0,
        "retweet_count": legacy.get("retweet_count") or 0,
        "reply_count": legacy.get("reply_count") or 0,
        "quote_count": legacy.get("quote_count") or 0,
        "created_at": str(legacy.get("created_at") or "").strip(),
        "url": url,
    }


def _candidate_handles(query: str) -> list[str]:
    """Extract probable account handles from query text.

    Prefer explicit @mentions. Fallback lexical tokens are only used when there
    are no explicit handles.
    """
    handles = [match.lstrip("@") for match in re.findall(r"@[A-Za-z0-9_]{2,20}", query)]
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9_]+", query)
        if 3 <= len(token) <= 20 and token.lower() not in X_HANDLE_STOPWORDS
    ]
    if handles:
        tokens = []
    combined: list[str] = []
    seen: set[str] = set()
    for value in handles + tokens:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(value)
        if len(combined) >= 8:
            break
    return combined


def _cache_key(project_id: str, query: str, max_results: int) -> str:
    """Generate a deterministic, project-scoped cache key for X searches."""
    query_hash = hashlib.sha256(f"{query}:{max_results}".encode()).hexdigest()[:16]
    return f"search:cache:{project_id}:social_x:posts:{query_hash}"


def _endpoint_candidates(path: str) -> list[str]:
    """Return ordered unique endpoint candidates for one path."""
    normalized_path = str(path or "").strip()
    path_variants = [normalized_path]
    if normalized_path.startswith("/twitter/"):
        path_variants.append(normalized_path.replace("/twitter/", "/x/", 1))
    candidates: list[str] = []
    for variant in path_variants:
        candidates.append(f"{settings.ENSEMBLE_API_BASE_URL.rstrip('/')}{variant}")
        candidates.append(f"{ENSEMBLE_BASE_FALLBACK.rstrip('/')}{variant}")
    out: list[str] = []
    seen: set[str] = set()
    for endpoint in candidates:
        if endpoint in seen:
            continue
        seen.add(endpoint)
        out.append(endpoint)
    return out


async def _get_json_from_candidates(
    client: httpx.AsyncClient,
    *,
    path: str,
    params: dict,
) -> object | None:
    """Call candidate endpoints and return the first successful JSON payload."""
    for endpoint in _endpoint_candidates(path):
        try:
            response = await client.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json"},
            )
            if _is_auth_required_error(int(response.status_code), response.text):
                logger.warning(
                    "social_x.auth_required",
                    endpoint=endpoint,
                    path=path,
                    status_code=int(response.status_code),
                )
                raise _XAuthRequiredError(response.text[:200])
            response.raise_for_status()
            payload = response.json()
            if _is_auth_required_error(None, payload):
                logger.warning("social_x.auth_required", endpoint=endpoint, path=path)
                raise _XAuthRequiredError(str(payload)[:200])
            return payload
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "social_x.endpoint_failed",
                endpoint=endpoint,
                path=path,
                status_code=int(exc.response.status_code),
                response_preview=str(exc.response.text or "")[:200],
            )
            continue
        except _XAuthRequiredError:
            raise
        except Exception as exc:
            logger.warning(
                "social_x.endpoint_failed",
                endpoint=endpoint,
                path=path,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            continue
    return None


async def _fetch_tweets_by_handle(client: httpx.AsyncClient, handle: str) -> list[dict]:
    """Resolve handle to user id, then fetch recent tweets."""
    info_payload = await _get_json_from_candidates(
        client,
        path="/twitter/user/info",
        params={"name": handle, "token": settings.ENSEMBLE_API_TOKEN},
    )
    if info_payload is None:
        return []

    info_dict = info_payload.get("data") if isinstance(info_payload, dict) else {}
    if not isinstance(info_dict, dict):
        info_dict = {}
    user_id = _normalize_user_id(
        info_dict.get("rest_id") or info_dict.get("user_id") or info_dict.get("id")
    )
    if not user_id:
        return []

    try:
        payload = await _get_json_from_candidates(
            client,
            path="/twitter/user/tweets",
            params={"id": user_id, "token": settings.ENSEMBLE_API_TOKEN},
        )
        if payload is None:
            return []
        return _extract_items(payload)
    except Exception:
        return []


async def search_x_posts(
    project_id: str, query: str, max_results: int = 20, redis=None
) -> list[dict]:
    """Retrieve X posts most relevant to project query with caching."""
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
            logger.warning("social_x.cache_read_failed", project_id=project_id)

    try:
        results = await _search_x_posts_impl(cleaned_query, max_results)
        if redis and results:
            try:
                await redis.setex(cache_key, settings.CACHE_TTL_SOCIAL, json.dumps(results))
                await redis.setex(stale_key, settings.CACHE_TTL_STALE, json.dumps(results))
            except Exception:
                logger.warning("social_x.cache_write_failed", project_id=project_id)
        return results
    except Exception as exc:
        if redis:
            try:
                stale = await redis.get(stale_key)
                if stale:
                    logger.info("social_x.serving_stale", project_id=project_id, error=str(exc))
                    return json.loads(stale)
            except Exception:
                pass
        raise


async def _search_x_posts_impl(query: str, max_results: int = 20) -> list[dict]:
    """Internal implementation of X post retrieval."""
    cleaned_query = query
    if not cleaned_query:
        return []

    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_x.no_token")
        return []

    bounded_max = max(1, min(max_results, X_MAX_RESULTS))
    logger.info(
        "social_x.search.start",
        query_preview=cleaned_query[:80],
        max_results=bounded_max,
    )

    posts: list[dict] = []
    async with httpx.AsyncClient(timeout=X_TIMEOUT_SECONDS) as client:
        # Try keyword-style endpoints first (if available for tenant plan).
        keyword_candidates = (
            (
                "/twitter/search",
                {
                    "name": cleaned_query,
                    "cursor": "",
                    "type": "latest",
                    "token": settings.ENSEMBLE_API_TOKEN,
                    **_keyword_auth_params(),
                },
            ),
            (
                "/twitter/keyword/search",
                {
                    "name": cleaned_query,
                    "cursor": "",
                    "type": "latest",
                    "token": settings.ENSEMBLE_API_TOKEN,
                    **_keyword_auth_params(),
                },
            ),
        )
        auth_required = False
        try:
            for path, params in keyword_candidates:
                payload = await _get_json_from_candidates(client, path=path, params=params)
                if payload is None:
                    continue
                candidate_posts = _extract_items(payload)
                if candidate_posts:
                    posts.extend(candidate_posts)
                    break
        except _XAuthRequiredError:
            auth_required = True
            logger.warning(
                "social_x.search.auth_required_skip_keyword", query_preview=cleaned_query[:80]
            )

        if not posts:
            handles = _candidate_handles(cleaned_query)
            if auth_required and not handles:
                return []
            for handle in handles:
                handle_posts = await _fetch_tweets_by_handle(client, handle)
                if handle_posts:
                    posts.extend(handle_posts)
                if len(posts) >= bounded_max * 3:
                    break

    results: list[dict] = []
    seen_ids: set[str] = set()
    for post in posts:
        if not isinstance(post, dict):
            continue
        normalized = _normalize_tweet_row(post)
        source_id = str(normalized.get("tweet_id") or normalized.get("id") or "").strip()
        content = str(normalized.get("text") or normalized.get("full_text") or "").strip()
        if not source_id or source_id in seen_ids or not content:
            continue
        seen_ids.add(source_id)
        username = str(
            normalized.get("screen_name")
            or normalized.get("username")
            or normalized.get("author_username")
            or ""
        ).strip()
        url = str(normalized.get("url") or normalized.get("tweet_url") or "").strip()
        if not url and username:
            url = f"https://x.com/{username}/status/{source_id}"
        if not url:
            url = f"https://x.com/i/web/status/{source_id}"

        results.append(
            {
                "source_id": source_id,
                "content": content,
                "author": username
                or str(normalized.get("name") or normalized.get("author") or "").strip(),
                "likes": _as_int(normalized.get("favorite_count") or normalized.get("likes")),
                "retweets": _as_int(normalized.get("retweet_count") or normalized.get("retweets")),
                "replies": _as_int(normalized.get("reply_count") or normalized.get("replies")),
                "quotes": _as_int(normalized.get("quote_count") or normalized.get("quotes")),
                "published_at": str(
                    normalized.get("create_time") or normalized.get("created_at") or ""
                ).strip(),
                "url": url,
            }
        )
        if len(results) >= bounded_max:
            break

    logger.info("social_x.search.success", query_preview=cleaned_query[:80], count=len(results))
    return results
