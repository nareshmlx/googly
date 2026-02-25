"""PatentsView API tool for patent search.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

PatentsView provides free, comprehensive access to USPTO patent data.
No API key required. Conservative rate limit: 5 req/sec (no official limit documented).

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time
from itertools import zip_longest

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, patentsview_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import patentsview_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.patentsview.org/patents/query"
_TIMEOUT = 15.0  # seconds
_MAX_RESULTS = 10

# Module-level HTTP client pool (reused across requests to prevent memory leak)
_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client pool for module."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _http_client


def _cache_key(query: str) -> str:
    """Generate deterministic cache key for a query."""
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"{CacheKeys.PATENTS_PATENTSVIEW}:{query_hash}"


async def _search_with_retry(query: str) -> dict | None:
    """
    Internal POST helper for the PatentsView /patents/query endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).
    """

    async def _fetch() -> dict:
        client = await _get_http_client()
        response = await client.post(
            _BASE_URL,
            json={
                "q": {"_text_all": {"patent_title": query}},
                "f": [
                    "patent_number",
                    "patent_title",
                    "patent_abstract",
                    "patent_date",
                    "inventor_first_name",
                    "inventor_last_name",
                ],
                "o": {"per_page": _MAX_RESULTS},
            },
        )
        response.raise_for_status()
        return response.json()

    # Wrap in rate limiter + retry
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            patentsview_limiter,
            "patentsview",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of PatentsView search (without cache/circuit breaker).

    Fetches from PatentsView API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query)
        api_calls_total.labels(api_name="patentsview", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="patentsview", status="error").inc()
        logger.error(
            "patents_patentsview.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="patentsview", status="timeout").inc()
        logger.warning(
            "patents_patentsview.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except Exception:
        api_calls_total.labels(api_name="patentsview", status="error").inc()
        logger.exception("patents_patentsview.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="patentsview").observe(duration)

    if data is None:
        logger.warning("patents_patentsview.no_data", query_preview=query[:80])
        return []

    patents = data.get("patents", [])
    if not patents or not isinstance(patents, list):
        logger.info("patents_patentsview.empty_results", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for p in patents:
        try:
            # Extract patent number and construct Google Patents URL
            patent_number = p.get("patent_number", "")
            google_patents_url = (
                f"https://patents.google.com/patent/US{patent_number}" if patent_number else ""
            )

            # Combine inventor names (handle multiple inventors)
            inventors = []
            first_names = p.get("inventor_first_name", [])
            last_names = p.get("inventor_last_name", [])

            # Handle both single inventor (non-list) and multiple inventors (list)
            if not isinstance(first_names, list):
                first_names = [first_names] if first_names else []
            if not isinstance(last_names, list):
                last_names = [last_names] if last_names else []

            # Combine first and last names (use zip_longest to avoid data loss from mismatched lengths)
            for first, last in zip_longest(first_names, last_names, fillvalue=""):
                first_str = first or ""
                last_str = last or ""
                if first_str or last_str:
                    inventors.append({"first_name": first_str, "last_name": last_str})

            normalized.append(
                {
                    "title": p.get("patent_title", ""),
                    "url": google_patents_url,
                    "content": p.get("patent_abstract", ""),
                    "patent_number": patent_number,
                    "date": p.get("patent_date", ""),
                    "inventors": inventors,
                    "source": "patentsview",
                }
            )
        except Exception:
            logger.exception("patents_patentsview.result_normalization_error", patent=p)
            continue

    logger.info("patents_patentsview.success", result_count=len(normalized))
    return normalized


async def _search_patentsview_impl(query: str, cache_key: str) -> list[dict]:
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
        logger.info("patents_patentsview.cache.hit", result_count=len(cached.get("results", [])))
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query)

    result = await call_with_circuit_breaker(
        patentsview_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config - patents don't change frequently)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_PATENTS)

    return result


async def search_patentsview(query: str) -> list[dict]:
    """
    Search for patents using PatentsView API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (5 req/sec conservative limit)
    - Retry with exponential backoff

    PatentsView provides free access to comprehensive USPTO patent data.
    No API key required.

    Request deduplication prevents thundering herd when multiple users
    search for the same patents simultaneously.

    Returned keys per result:
      - title (str):          Patent title
      - url (str):            Google Patents URL (https://patents.google.com/patent/US{number})
      - content (str):        Patent abstract
      - patent_number (str):  USPTO patent number
      - date (str):           Patent grant date (YYYY-MM-DD)
      - inventors (list[str]): List of inventor names (first + last)
      - source (str):         Always "patentsview"

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Sanitize and normalize query BEFORE cache key generation (prevents cache key collision)
    query = query.strip().lower()
    if not query:
        logger.warning("patents_patentsview.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("patents_patentsview.query_too_long", original_length=len(query))
        query = query[:1000]

    logger.info("patents_patentsview.start", query_preview=query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_patentsview_impl(query, cache_key),
    )

    logger.info("patents_patentsview.complete", result_count=len(result))
    return result
