"""Tavily web search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

Tavily provides LLM-optimized web search with clean content extraction.
Growth plan: 10 req/sec, $0.006 per request.

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, tavily_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import rate_limited_call, tavily_limiter
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_TAVILY_BASE_URL = "https://api.tavily.com/search"
_TIMEOUT = 15.0  # seconds
_MAX_RESULTS = 5


def _cache_key(query: str) -> str:
    """Generate deterministic cache key for a query."""
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"{CacheKeys.SEARCH_TAVILY}:{query_hash}"


async def _search_with_retry(query: str) -> dict | None:
    """
    Internal POST helper for the Tavily /search endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).
    """

    async def _fetch() -> dict:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                _TAVILY_BASE_URL,
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": _MAX_RESULTS,
                    "include_answer": False,  # We synthesize our own answer
                    "include_raw_content": False,
                    "include_images": False,
                },
            )
            response.raise_for_status()
            return response.json()

    # Wrap in rate limiter + retry
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            tavily_limiter,
            "tavily",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of Tavily search (without cache/circuit breaker).

    Fetches from Tavily API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query)
        api_calls_total.labels(api_name="tavily", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="tavily", status="error").inc()
        logger.error(
            "search_tavily.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="tavily", status="timeout").inc()
        logger.warning(
            "search_tavily.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except Exception:
        api_calls_total.labels(api_name="tavily", status="error").inc()
        logger.exception("search_tavily.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="tavily").observe(duration)

    if data is None:
        logger.warning("search_tavily.no_data", query_preview=query[:80])
        return []

    results = data.get("results", [])
    if not results or not isinstance(results, list):
        logger.info("search_tavily.empty_results", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for r in results:
        try:
            normalized.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "source": "tavily",
                    "published_date": r.get("published_date"),
                }
            )
        except Exception:
            logger.exception("search_tavily.result_normalization_error", result=r)
            continue

    logger.info("search_tavily.success", result_count=len(normalized))
    return normalized


async def _search_tavily_impl(query: str, cache_key: str) -> list[dict]:
    """
    Internal implementation (called by dedup layer).

    Handles cache check, API call, and cache write.
    """
    # Initialize cache (lazy singleton pattern)
    redis_client = await get_redis()
    cache = TwoTierCache(redis_client)

    # Check cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        logger.info("search_tavily.cache.hit", result_count=len(cached.get("results", [])))
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query)

    result = await call_with_circuit_breaker(
        tavily_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config for web search)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_SEARCH)

    return result


async def search_tavily(query: str) -> list[dict]:
    """
    Search the web using Tavily API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (10 req/sec for Growth plan)
    - Retry with exponential backoff

    Tavily provides LLM-optimized web search with clean content extraction.
    Growth plan: 10 req/sec, $0.006 per request.

    Returned keys per result:
      - title (str):          Page title
      - url (str):            Page URL
      - content (str):        Extracted clean content (LLM-optimized)
      - score (float):        Relevance score (0.0-1.0)
      - source (str):         Always "tavily"
      - published_date (str): Publication date if available

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Validate API key is set
    if not settings.TAVILY_API_KEY:
        logger.error("search_tavily.api_key_missing")
        return []

    # Sanitize query
    query = query.strip()
    if not query:
        logger.warning("search_tavily.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("search_tavily.query_too_long", original_length=len(query))
        query = query[:1000]

    logger.info("search_tavily.start", query_preview=query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_tavily_impl(query, cache_key),
    )

    logger.info("search_tavily.complete", result_count=len(result))
    return result
