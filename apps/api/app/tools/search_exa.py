"""Exa semantic search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

Exa provides neural/semantic search optimized for company intelligence and research.
Growth plan: 5 req/sec, $0.002 per request (search), $0.006 per request (with contents).

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, exa_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import exa_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_EXA_BASE_URL = "https://api.exa.ai/search"
_TIMEOUT = 15.0  # seconds
_NUM_RESULTS = 10
_MAX_CHARACTERS = 1000
# Domain-filtered queries extract more text per result — the user named a specific
# site and needs rich content (descriptions, notes, reviews), not short snippets.
_MAX_CHARACTERS_DOMAIN = 2000


def _cache_key(query: str, domains: list[str] | None = None) -> str:
    """Generate deterministic cache key for a query, optionally scoped to domains.

    Domain-filtered requests produce a different key than unfiltered ones so
    "techcrunch.com + retinol" never hits the cache entry for bare "retinol".
    """
    raw = query + str(sorted(domains or []))
    query_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{CacheKeys.SEARCH_EXA}:{query_hash}"


async def _search_with_retry(query: str, include_domains: list[str] | None = None) -> dict | None:
    """
    Internal POST helper for the Exa /search endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).

    When ``include_domains`` is provided, results are restricted to those domains.
    Exa uses the camelCase key ``includeDomains`` in its request body.
    """

    async def _fetch() -> dict:
        # Use more content per result for domain-filtered queries — the user named
        # a specific site, so deeper extraction (notes, ratings, descriptions)
        # produces much richer synthesis than the default snippet length.
        payload: dict = {
            "query": query,
            "type": "neural",  # Neural/semantic search
            "numResults": _NUM_RESULTS,
            "useAutoprompt": True,  # Let Exa optimize query
            "contents": {
                "text": {
                    "maxCharacters": _MAX_CHARACTERS_DOMAIN if include_domains else _MAX_CHARACTERS,
                }
            },
        }
        if include_domains:
            payload["includeDomains"] = include_domains
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                _EXA_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.EXA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    # Wrap in rate limiter + retry
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            exa_limiter,
            "exa",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _search_impl(query: str, include_domains: list[str] | None = None) -> list[dict]:
    """
    Internal implementation of Exa search (without cache/circuit breaker).

    Fetches from Exa API and normalizes results into consistent dict format.
    When ``include_domains`` is provided, only results from those domains are returned.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query, include_domains=include_domains)
        api_calls_total.labels(api_name="exa", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="exa", status="error").inc()
        logger.error(
            "search_exa.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="exa", status="timeout").inc()
        logger.warning(
            "search_exa.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except Exception:
        api_calls_total.labels(api_name="exa", status="error").inc()
        logger.exception("search_exa.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="exa").observe(duration)

    if data is None:
        logger.warning("search_exa.no_data", query_preview=query[:80])
        return []

    results = data.get("results", [])
    if not results or not isinstance(results, list):
        logger.info("search_exa.empty_results", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for r in results:
        try:
            # Exa returns results with text content in the 'text' field.
            # Use the domain-appropriate character limit so domain-filtered queries
            # (which request 2000 chars from Exa) are not silently truncated back to 1000.
            text_content = ""
            if isinstance(r.get("text"), str):
                char_limit = _MAX_CHARACTERS_DOMAIN if include_domains else _MAX_CHARACTERS
                text_content = r["text"][:char_limit]

            normalized.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": text_content,
                    "score": r.get("score", 0.0),
                    "source": "exa",
                    "published_date": r.get("publishedDate"),
                    # Emit as `authors` list (consistent with all other tools).
                    # Exa returns a single string or None under `author`.
                    "authors": [r["author"]] if r.get("author") else [],
                }
            )
        except Exception:
            logger.exception("search_exa.result_normalization_error", result=r)
            continue

    logger.info("search_exa.success", result_count=len(normalized))
    return normalized


async def _search_exa_impl(
    query: str, cache_key: str, include_domains: list[str] | None = None
) -> list[dict]:
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
        logger.info("search_exa.cache.hit", result_count=len(cached.get("results", [])))
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query, include_domains=include_domains)

    result = await call_with_circuit_breaker(
        exa_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config for web search)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_SEARCH)

    return result


async def search_exa(query: str, include_domains: list[str] | None = None) -> list[dict]:
    """
    Search using Exa semantic/neural search API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (5 req/sec for Growth plan)
    - Retry with exponential backoff

    Exa provides neural/semantic search optimized for company intelligence and research.
    Growth plan: 5 req/sec, $0.006 per request (with contents).

    Args:
        query:           Search query string (max 1000 chars).
        include_domains: Optional list of domains to restrict results to
                         (e.g. ["techcrunch.com"]).  None means no restriction.
                         Domain-filtered calls use a separate cache key so they
                         never collide with unfiltered results for the same query.
                         Passed to Exa as ``includeDomains`` (camelCase).

    Returned keys per result:
      - title (str):          Page title
      - url (str):            Page URL
      - content (str):        Extracted content (max 1000 chars)
      - score (float):        Relevance score
      - source (str):         Always "exa"
      - published_date (str): Publication date if available
      - authors (list[str]):  Author list (empty list if unavailable)

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Validate API key is set
    if not settings.EXA_API_KEY:
        logger.error("search_exa.api_key_missing")
        return []

    # Sanitize query
    query = query.strip()
    if not query:
        logger.warning("search_exa.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("search_exa.query_too_long", original_length=len(query))
        query = query[:1000]

    logger.info(
        "search_exa.start",
        query_preview=query[:80],
        include_domains=include_domains,
    )

    # Generate cache key (used for both dedup and cache); domains are included
    # so domain-filtered queries never share a cache entry with bare queries.
    cache_key = _cache_key(query, domains=include_domains)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_exa_impl(query, cache_key, include_domains=include_domains),
    )

    logger.info("search_exa.complete", result_count=len(result))
    return result
