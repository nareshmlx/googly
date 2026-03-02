"""Reddit retrieval tool backed by EnsembleData subreddit-posts endpoint."""

import hashlib
import json
import re

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

REDDIT_MAX_POSTS_PER_SUBREDDIT = 40  # Increased from 10 to allow more posts per subreddit
REDDIT_TIMEOUT_SECONDS = 20.0
REDDIT_MIN_TOKEN_LEN = 3
REDDIT_TOKEN_MAX_LEN = 40
ENSEMBLE_BASE_FALLBACK = "https://ensembledata.com/apis"
REDDIT_STOPWORDS = frozenset(
    {
        "about",
        "analysis",
        "best",
        "beauty",
        "current",
        "fashion",
        "find",
        "for",
        "from",
        "latest",
        "need",
        "news",
        "on",
        "post",
        "posts",
        "reddit",
        "research",
        "show",
        "source",
        "tell",
        "the",
        "this",
        "trend",
        "trends",
        "what",
        "with",
    }
)
TOPIC_SUBREDDIT_HINTS: dict[str, str] = {
    "skincare": "SkincareAddiction",
    "skin": "SkincareAddiction",
    "makeup": "MakeupAddiction",
    "cosmetic": "beauty",
    "cosmetics": "beauty",
    "beauty": "beauty",
    "fashion": "femalefashionadvice",
    "fragrance": "fragrance",
    "perfume": "fragrance",
    "hair": "HaircareScience",
}


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


def _curated_subreddits() -> list[str]:
    """Parse curated base subreddits from settings."""
    raw = str(settings.INGEST_REDDIT_BASE_SUBREDDITS or "")
    values: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", token).strip()
        if len(cleaned) < REDDIT_MIN_TOKEN_LEN or len(cleaned) > REDDIT_TOKEN_MAX_LEN:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(cleaned)
    return values


def _dynamic_subreddit_candidates(query: str, max_dynamic: int) -> list[str]:
    """Extract bounded subreddit candidates from explicit mentions and topical hints."""
    explicit_mentions = list(
        re.findall(r"(?:^|\s)r/([A-Za-z0-9_]{3,40})", query, flags=re.IGNORECASE)
    )

    candidates: list[str] = []
    seen: set[str] = set()
    for token in explicit_mentions:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", token).strip()
        key = cleaned.lower()
        if not cleaned or key in seen or key in REDDIT_STOPWORDS:
            continue
        seen.add(key)
        candidates.append(cleaned)
        if len(candidates) >= max_dynamic:
            return candidates
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        hinted = TOPIC_SUBREDDIT_HINTS.get(token.lower())
        if not hinted:
            continue
        key = hinted.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(hinted)
        if len(candidates) >= max_dynamic:
            return candidates
    return candidates


def _candidate_subreddits(query: str) -> list[str]:
    """Build subreddit call list from curated allowlist plus bounded dynamic expansion."""
    curated = _curated_subreddits()
    dynamic = _dynamic_subreddit_candidates(
        query,
        max_dynamic=int(settings.INGEST_REDDIT_MAX_DYNAMIC_SUBREDDITS or 0),
    )

    candidates: list[str] = []
    seen: set[str] = set()
    for name in curated + dynamic:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(name)
    return candidates


def _extract_items(payload: object) -> list[dict]:
    """Extract list records from common API envelope shapes."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    candidates = [payload.get("data"), payload.get("posts"), payload.get("results")]
    for value in candidates:
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for nested_key in ("data", "posts", "results"):
                nested = value.get(nested_key)
                if isinstance(nested, list):
                    return [row for row in nested if isinstance(row, dict)]
    return []


def _normalize_row(row: dict) -> dict:
    """Flatten Reddit wrappers like {'kind':'t3','data':{...}}."""
    nested = row.get("data")
    if isinstance(nested, dict):
        return nested
    return row


def _extract_next_cursor(payload: object) -> str:
    """Extract pagination cursor from common response envelope shapes."""
    if not isinstance(payload, dict):
        return ""
    cursor_candidates = (
        payload.get("nextCursor"),
        payload.get("next_cursor"),
        payload.get("cursor"),
    )
    for value in cursor_candidates:
        if value is not None and str(value).strip():
            return str(value).strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("nextCursor", "next_cursor", "cursor"):
            value = data.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _cache_key(project_id: str, query: str, limit: int, time_filter: str) -> str:
    """Generate a deterministic, project-scoped cache key for Reddit searches."""
    query_hash = hashlib.sha256(f"{query}:{limit}:{time_filter}".encode()).hexdigest()[:16]
    return f"search:cache:{project_id}:social_reddit:posts:{query_hash}"


def _endpoint_candidates() -> list[str]:
    """Return ordered unique candidate endpoints for subreddit posts."""
    candidates = [
        f"{settings.ENSEMBLE_API_BASE_URL.rstrip('/')}/reddit/subreddit/posts",
        f"{ENSEMBLE_BASE_FALLBACK.rstrip('/')}/reddit/subreddit/posts",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for endpoint in candidates:
        if endpoint in seen:
            continue
        seen.add(endpoint)
        out.append(endpoint)
    return out


async def search_reddit_posts(
    project_id: str, query: str, limit: int = 24, time_filter: str = "week", redis=None
) -> list[dict]:
    """Aggregate top Reddit posts from intent-derived subreddits with caching."""
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return []

    # Check cache first
    cache_key = _cache_key(project_id, cleaned_query, limit, time_filter)
    stale_key = f"{cache_key}:stale"
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            logger.warning("social_reddit.cache_read_failed", project_id=project_id)

    try:
        results = await _search_reddit_posts_impl(cleaned_query, limit, time_filter)
        if redis and results:
            try:
                await redis.setex(cache_key, settings.CACHE_TTL_SOCIAL, json.dumps(results))
                await redis.setex(stale_key, settings.CACHE_TTL_STALE, json.dumps(results))
            except Exception:
                logger.warning("social_reddit.cache_write_failed", project_id=project_id)
        return results
    except Exception as exc:
        if redis:
            try:
                stale = await redis.get(stale_key)
                if stale:
                    logger.info(
                        "social_reddit.serving_stale", project_id=project_id, error=str(exc)
                    )
                    return json.loads(stale)
            except Exception:
                pass
        raise


async def _search_reddit_posts_impl(
    query: str, limit: int = 24, time_filter: str = "week"
) -> list[dict]:
    """Internal implementation of Reddit post retrieval."""
    cleaned_query = query
    if not cleaned_query:
        return []

    if not settings.ENSEMBLE_API_TOKEN:
        logger.warning("social_reddit.no_token")
        return []

    curated = _curated_subreddits()
    subreddits = _candidate_subreddits(cleaned_query)
    if not subreddits:
        return []

    per_subreddit_limit = max(1, min(REDDIT_MAX_POSTS_PER_SUBREDDIT, max(20, limit)))
    logger.info(
        "social_reddit.search.start",
        query_preview=cleaned_query[:80],
        subreddits=subreddits,
        per_subreddit_limit=per_subreddit_limit,
        curated_count=len(curated),
        dynamic_limit=int(settings.INGEST_REDDIT_MAX_DYNAMIC_SUBREDDITS or 0),
    )

    rows: list[dict] = []
    endpoints = _endpoint_candidates()
    max_pages = max(1, int(settings.INGEST_REDDIT_MAX_PAGES_PER_SUBREDDIT))
    async with httpx.AsyncClient(timeout=REDDIT_TIMEOUT_SECONDS) as client:
        for subreddit in subreddits:
            cursor = ""
            for _ in range(max_pages):
                params = {
                    "name": subreddit,
                    "sort": "top",
                    "period": time_filter,
                    "cursor": cursor,
                    "token": settings.ENSEMBLE_API_TOKEN,
                }
                payload: object | None = None
                for endpoint in endpoints:
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
                            "social_reddit.subreddit_endpoint_failed",
                            subreddit=subreddit,
                            endpoint=endpoint,
                            status_code=int(exc.response.status_code),
                            response_preview=str(exc.response.text or "")[:200],
                        )
                        continue
                    except Exception as exc:
                        logger.warning(
                            "social_reddit.subreddit_endpoint_failed",
                            subreddit=subreddit,
                            endpoint=endpoint,
                            error_type=type(exc).__name__,
                            error=str(exc)[:200],
                        )
                        continue
                if payload is None:
                    logger.warning("social_reddit.subreddit_fetch_failed", subreddit=subreddit)
                    continue
                fetched = _extract_items(payload)
                if fetched:
                    rows.extend(fetched[:per_subreddit_limit])
                if len(rows) >= limit:
                    break
                next_cursor = _extract_next_cursor(payload)
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
            if len(rows) >= limit:
                break

    query_terms = {
        token.lower() for token in re.findall(r"[a-z0-9]+", cleaned_query) if len(token) >= 3
    }
    filtered_rows: list[dict] = []
    for row in rows:
        normalized = _normalize_row(row)
        combined = f"{normalized.get('title', '')} {normalized.get('selftext', '')}".lower()
        if query_terms and not any(term in combined for term in query_terms):
            continue
        filtered_rows.append(normalized)

    deduped: list[dict] = []
    seen_ids: set[str] = set()
    for row in filtered_rows:
        source_id = str(
            row.get("id") or row.get("post_id") or row.get("name") or row.get("fullname") or ""
        ).strip()
        if source_id.startswith("t3_"):
            source_id = source_id.removeprefix("t3_")
        if not source_id or source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        permalink = str(row.get("permalink") or row.get("post_url") or "").strip()
        url = str(row.get("url") or row.get("post_url") or "").strip()
        if permalink and not permalink.startswith("http"):
            permalink = f"https://www.reddit.com{permalink}"
        subreddit_name = str(row.get("subreddit_name") or row.get("subreddit") or "").strip()
        if not permalink and not url and subreddit_name and source_id:
            permalink = f"https://www.reddit.com/r/{subreddit_name}/comments/{source_id}"
        deduped.append(
            {
                "source_id": source_id,
                "title": str(row.get("title") or "").strip(),
                "content": str(row.get("selftext") or "").strip(),
                "author": str(row.get("author_username") or row.get("author") or "").strip(),
                "subreddit": subreddit_name,
                "score": _as_int(row.get("score")),
                "comments": _as_int(row.get("num_comments") or row.get("comment_count")),
                "published_at": row.get("create_time") or row.get("created_utc") or "",
                "url": permalink or url,
            }
        )
        if len(deduped) >= limit:
            break

    logger.info(
        "social_reddit.search.success", query_preview=cleaned_query[:80], count=len(deduped)
    )
    return deduped
