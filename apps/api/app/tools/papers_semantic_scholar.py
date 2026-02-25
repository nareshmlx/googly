"""Semantic Scholar academic paper search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

Semantic Scholar provides access to 200M+ academic papers from all fields of science.
Free tier: 1 req/sec, no API key required (optional for higher limits).

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, semantic_scholar_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import rate_limited_call, semantic_scholar_limiter
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
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
    return f"{CacheKeys.PAPERS_SEMANTIC_SCHOLAR}:{query_hash}"


async def _search_with_retry(query: str) -> dict | None:
    """
    Internal GET helper for the Semantic Scholar /paper/search endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).
    """

    async def _fetch() -> dict:
        client = await _get_http_client()
        # Build headers with optional API key
        headers = {}
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

        response = await client.get(
            _BASE_URL,
            params={
                "query": query,
                "limit": _MAX_RESULTS,
                "fields": "paperId,title,authors,year,abstract,url,citationCount",
            },
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    # Wrap in rate limiter + retry
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            semantic_scholar_limiter,
            "semantic_scholar",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of Semantic Scholar search (without cache/circuit breaker).

    Fetches from Semantic Scholar API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query)
        api_calls_total.labels(api_name="semantic_scholar", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="semantic_scholar", status="error").inc()
        logger.error(
            "search_semantic_scholar.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="semantic_scholar", status="timeout").inc()
        logger.warning(
            "search_semantic_scholar.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except Exception:
        api_calls_total.labels(api_name="semantic_scholar", status="error").inc()
        logger.exception("search_semantic_scholar.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="semantic_scholar").observe(duration)

    if data is None:
        logger.warning("search_semantic_scholar.no_data", query_preview=query[:80])
        return []

    papers = data.get("data", [])
    if not papers or not isinstance(papers, list):
        logger.info("search_semantic_scholar.empty_results", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for paper in papers:
        try:
            # Extract author names from authors array
            authors = paper.get("authors", [])
            author_names = [
                author.get("name", "") for author in authors if isinstance(author, dict)
            ]

            normalized.append(
                {
                    "title": paper.get("title", ""),
                    "url": paper.get("url", ""),
                    "content": paper.get("abstract", ""),  # Use abstract as content
                    "authors": author_names,
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount", 0),
                    "source": "semantic_scholar",
                    "paper_id": paper.get("paperId", ""),
                }
            )
        except Exception:
            logger.exception("search_semantic_scholar.result_normalization_error", result=paper)
            continue

    logger.info("search_semantic_scholar.success", result_count=len(normalized))
    return normalized


async def _search_semantic_scholar_impl(query: str, cache_key: str) -> list[dict]:
    """
    Internal implementation (called by dedup layer).

    Handles cache check, API call, and cache write.
    """
    # Initialize cache (lazy singleton pattern)
    redis_client = await get_redis()
    cache = TwoTierCache(redis_client)

    # Check cache first (using normalized query)
    cached = await cache.get(cache_key)
    if cached is not None:
        logger.info(
            "search_semantic_scholar.cache.hit", result_count=len(cached.get("results", []))
        )
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query)

    result = await call_with_circuit_breaker(
        semantic_scholar_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config - research papers don't change frequently)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_PAPERS)

    return result


async def search_semantic_scholar(query: str) -> list[dict]:
    """
    Search academic papers using Semantic Scholar API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results (TTL from config)
    - Circuit breaker protection for API failures
    - Rate limiting (1 req/sec for free tier)
    - Retry with exponential backoff

    Semantic Scholar provides access to 200M+ academic papers from all fields of science.
    Free tier: 1 req/sec, no API key required (optional for higher limits).

    Returned keys per result:
      - title (str):          Paper title
      - url (str):            Paper URL on Semantic Scholar
      - content (str):        Paper abstract (used as content)
      - authors (list[str]):  List of author names
      - year (int):           Publication year
      - citation_count (int): Number of citations
      - source (str):         Always "semantic_scholar"
      - paper_id (str):       Semantic Scholar paper ID

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # API key is optional for free tier, but log if missing
    if not settings.SEMANTIC_SCHOLAR_API_KEY:
        logger.debug("search_semantic_scholar.no_api_key", message="Using free tier limits")

    # Sanitize and normalize query BEFORE cache key generation (prevents cache key collision)
    query = query.strip().lower()
    if not query:
        logger.warning("search_semantic_scholar.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("search_semantic_scholar.query_too_long", original_length=len(query))
        query = query[:1000]

    logger.info("search_semantic_scholar.start", query_preview=query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_semantic_scholar_impl(query, cache_key),
    )

    logger.info("search_semantic_scholar.complete", result_count=len(result))
    return result
