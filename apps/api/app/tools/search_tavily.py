"""Tavily web search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

Tavily provides LLM-optimized web search with clean content extraction.
Growth plan: 10 req/sec, $0.006 per request.

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import json
import time

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, tavily_breaker
from app.core.config import settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import rate_limited_call, tavily_limiter
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_TAVILY_BASE_URL = "https://api.tavily.com/search"
_TIMEOUT = 15.0  # seconds
_MAX_RESULTS = 15
# Domain-filtered queries use advanced depth (full page extraction vs snippets) and
# request more results, because the user named a specific site and needs rich detail.
_MAX_RESULTS_DOMAIN = 7
_SEARCH_DEPTH_DOMAIN = "advanced"


def _cache_key(project_id: str, query: str, domains: list[str] | None = None) -> str:
    """Generate deterministic project-scoped cache key for a query."""
    raw = query + str(sorted(domains or []))
    query_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"search:cache:{project_id}:tavily:search:{query_hash}"


async def _search_with_retry(query: str, include_domains: list[str] | None = None) -> dict | None:
    """
    Internal POST helper for the Tavily /search endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).

    When ``include_domains`` is provided, results are restricted to those domains.
    """

    async def _fetch() -> dict:
        # Use advanced depth + more results for domain-filtered queries — the user
        # explicitly named a site, so we crawl deeper to get full page content
        # (notes, ratings, descriptions) rather than short snippets.
        payload: dict = {
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": _MAX_RESULTS_DOMAIN if include_domains else _MAX_RESULTS,
            "include_answer": False,  # We synthesize our own answer
            "include_raw_content": False,
            "include_images": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(_TAVILY_BASE_URL, json=payload)
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


async def _search_impl(query: str, include_domains: list[str] | None = None) -> list[dict]:
    """
    Internal implementation of Tavily search (without cache/circuit breaker).

    Fetches from Tavily API and normalizes results into consistent dict format.
    When ``include_domains`` is provided, only results from those domains are returned.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query, include_domains=include_domains)
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


async def _search_tavily_impl(
    project_id: str, query: str, cache_key: str, include_domains: list[str] | None = None
) -> list[dict]:
    """
    Internal implementation (called by dedup layer).
    Handles cache check, API call, and cache write with stale fallback.
    """
    redis_client = await get_redis()
    cache = TwoTierCache(redis_client)

    # 1. Primary Cache Check (Fresh result)
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached.get("results", [])

    # 2. Fetch from API
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query, include_domains=include_domains)

    result = await call_with_circuit_breaker(tavily_breaker, _fetch_impl)

    # 3. Success -> Cache and Return
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_SEARCH)
        # Store a "stale" copy that lasts longer
        stale_key = f"{cache_key}:stale"
        await redis_client.setex(
            stale_key, settings.CACHE_TTL_STALE, json.dumps({"results": result})
        )
        return result

    # 4. Failure -> Serve Stale Fallback
    stale_key = f"{cache_key}:stale"
    stale_raw = await redis_client.get(stale_key)
    if stale_raw:
        logger.warning(
            "search_tavily.api_failed.serving_stale", project_id=project_id, query=query[:50]
        )
        try:
            return json.loads(stale_raw).get("results", [])
        except json.JSONDecodeError:
            pass

    return []


async def search_tavily(
    project_id: str, query: str, include_domains: list[str] | None = None
) -> list[dict]:
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

    Args:
        query:           Search query string (max 1000 chars).
        include_domains: Optional list of domains to restrict results to
                         (e.g. ["techcrunch.com"]).  None means no restriction.
                         Domain-filtered calls use a separate cache key so they
                         never collide with unfiltered results for the same query.

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

    logger.info(
        "search_tavily.start",
        query_preview=query[:80],
        include_domains=include_domains,
    )

    # Generate cache key (instrumented with project_id for multi-tenant security)
    cache_key = _cache_key(project_id, query, domains=include_domains)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_tavily_impl(project_id, query, cache_key, include_domains=include_domains),
    )

    logger.info("search_tavily.complete", result_count=len(result))
    return result
