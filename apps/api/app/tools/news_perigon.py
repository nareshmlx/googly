"""Perigon news search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

Perigon provides news aggregation with sentiment analysis and story threading.
Plus plan: 4 req/sec, $0.011 per request (50k req/mo = $550).

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time
from datetime import UTC, datetime, timedelta

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, perigon_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import perigon_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_PERIGON_BASE_URL = "https://api.goperigon.com/v1/all"
_TIMEOUT = 15.0  # seconds
_CATEGORIES = "Business,Tech,Lifestyle"
_DAYS_LOOKBACK = 30


def _cache_key(query: str) -> str:
    """Generate deterministic cache key for a query."""
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"{CacheKeys.NEWS_PERIGON}:{query_hash}"


async def _search_with_retry(query: str) -> dict | None:
    """
    Internal GET helper for the Perigon /v1/all endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).
    """
    # Calculate date range (last 30 days)
    from_date = (datetime.now(UTC) - timedelta(days=_DAYS_LOOKBACK)).isoformat()

    async def _fetch() -> dict:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(
                _PERIGON_BASE_URL,
                params={
                    "apiKey": settings.PERIGON_API_KEY,
                    "q": query,
                    "sortBy": "relevance",
                    "showReprints": "false",
                    "showNumResults": "true",
                    "category": _CATEGORIES,
                    "from": from_date,
                },
            )
            response.raise_for_status()
            return response.json()

    # Wrap in rate limiter + retry
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            perigon_limiter,
            "perigon",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of Perigon news search (without cache/circuit breaker).

    Fetches from Perigon API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query)
        api_calls_total.labels(api_name="perigon", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="perigon", status="error").inc()
        logger.error(
            "news_perigon.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="perigon", status="timeout").inc()
        logger.warning(
            "news_perigon.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except Exception:
        api_calls_total.labels(api_name="perigon", status="error").inc()
        logger.exception("news_perigon.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="perigon").observe(duration)

    if data is None:
        logger.warning("news_perigon.no_data", query_preview=query[:80])
        return []

    articles = data.get("articles", [])
    if not articles or not isinstance(articles, list):
        logger.info("news_perigon.empty_results", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for article in articles:
        try:
            # Extract source name safely
            source_name = ""
            if isinstance(article.get("source"), dict):
                source_name = article["source"].get("name", "")

            normalized.append(
                {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "content": article.get("description", ""),
                    "source": "perigon",
                    "published_date": article.get("pubDate"),
                    "source_name": source_name,
                    "sentiment": article.get("sentiment"),  # Unique to Perigon
                    "entities": article.get("entities", []),  # Unique to Perigon
                    "story_id": article.get("storyId"),  # For threading (Phase 3)
                }
            )
        except Exception:
            logger.exception("news_perigon.result_normalization_error", article=article)
            continue

    logger.info("news_perigon.success", result_count=len(normalized))
    return normalized


async def _search_perigon_impl(query: str, cache_key: str) -> list[dict]:
    """
    Internal implementation (called by dedup layer).

    Handles cache check, API call, and cache write.
    This function is only called once even if 1000 users search
    for the same query simultaneously.
    """
    # Initialize cache (lazy singleton pattern)
    redis_client = await get_redis()
    cache = TwoTierCache(redis_client)

    # Check cache first (using normalized query)
    cached = await cache.get(cache_key)
    if cached is not None:
        logger.info("news_perigon.cache.hit", result_count=len(cached.get("results", [])))
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query)

    result = await call_with_circuit_breaker(
        perigon_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config for news - shorter than web search)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_NEWS)

    return result


async def search_perigon(query: str) -> list[dict]:
    """
    Search news articles using Perigon API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (4 req/sec for Plus plan)
    - Retry with exponential backoff

    Perigon provides news aggregation with sentiment analysis and story threading.
    Plus plan: 4 req/sec, $0.011 per request (50k req/mo = $550).

    Request deduplication prevents thundering herd when multiple users
    search for the same news simultaneously.

    Searches news from last 30 days in Business, Tech, and Lifestyle categories.

    Returned keys per result:
      - title (str):          Article title
      - url (str):            Article URL
      - content (str):        Article description/summary
      - source (str):         Always "perigon"
      - published_date (str): Publication date (ISO format)
      - source_name (str):    Original news source (e.g., "TechCrunch")
      - sentiment (dict):     Sentiment analysis (Perigon feature)
      - entities (list):      Extracted entities (Perigon feature)
      - story_id (str):       Story thread ID for related articles

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Validate API key is set
    if not settings.PERIGON_API_KEY:
        logger.error("news_perigon.api_key_missing")
        return []

    # Sanitize query
    query = query.strip()
    if not query:
        logger.warning("news_perigon.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("news_perigon.query_too_long", original_length=len(query))
        query = query[:1000]

    logger.info("news_perigon.start", query_preview=query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_perigon_impl(query, cache_key),
    )

    logger.info("news_perigon.complete", result_count=len(result))
    return result
