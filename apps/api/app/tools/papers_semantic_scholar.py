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
_MAX_RESULTS = 20

# Module-level HTTP client pool (reused across requests to prevent memory leak)
_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client pool for module."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _http_client


def _build_semantic_query(
    query: str,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
) -> str:
    """Compose a Semantic Scholar query that preserves specific user terms."""
    must_match_terms = [str(term).strip() for term in (must_match_terms or []) if str(term).strip()]
    domain_terms = [str(term).strip() for term in (domain_terms or []) if str(term).strip()]
    if not must_match_terms:
        return query
    # Use all must_match_terms with OR logic (no limit)
    parts = [f'"{term}"' for term in must_match_terms]
    parts.extend(f'"{term}"' for term in domain_terms)
    return " ".join(parts).strip()


def _cache_key(query: str) -> str:
    """Generate deterministic cache key for a query."""
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"{CacheKeys.PAPERS_SEMANTIC_SCHOLAR}:{query_hash}"


async def _search_with_retry(query: str) -> dict | None:
    """
    Internal GET helper for the Semantic Scholar /paper/search endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).

    retry_with_backoff never raises — it catches all exceptions internally and returns None.
    Callers must check the return value rather than catching exceptions from this function.
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

    async def _rate_limited_fetch() -> dict:
        """Named wrapper so retry logs emit func=_rate_limited_fetch, not func=<lambda>."""
        return await rate_limited_call(
            semantic_scholar_limiter,
            "semantic_scholar",
            _fetch,
        )

    # Wrap in rate limiter + retry.
    # 4xx errors (except 429) are permanent — skip retries immediately.
    # A 403 with no API key or an expired key must not burn 1s+2s of backoff sleep.
    # retry_with_backoff logs non-retryable errors at retry.non_retryable_http_error.
    result = await retry_with_backoff(
        _rate_limited_fetch,
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
        non_retryable_statuses=frozenset({400, 401, 403, 404, 410, 422}),
    )
    return result


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of Semantic Scholar search (without cache/circuit breaker).

    Fetches from Semantic Scholar API and normalizes results into consistent dict format.

    retry_with_backoff never raises — all error handling happens inside it and the result
    is None on failure. Metrics are only labelled "success" when data is actually returned.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query)
    except Exception:
        # Defensive catch — retry_with_backoff should never raise, but guard anyway.
        api_calls_total.labels(api_name="semantic_scholar", status="error").inc()
        logger.exception("search_semantic_scholar.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="semantic_scholar").observe(duration)

    if data is None:
        # retry_with_backoff returned None — all attempts failed (logged inside retry)
        api_calls_total.labels(api_name="semantic_scholar", status="error").inc()
        logger.warning("search_semantic_scholar.no_data", query_preview=query[:80])
        return []

    api_calls_total.labels(api_name="semantic_scholar", status="success").inc()

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


async def search_semantic_scholar(
    query: str,
    *,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
    query_specificity: str | None = None,
) -> list[dict]:
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
    query = query.strip()
    if not query:
        logger.warning("search_semantic_scholar.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("search_semantic_scholar.query_too_long", original_length=len(query))
        query = query[:1000]

    effective_query = _build_semantic_query(query, must_match_terms, domain_terms)
    if str(query_specificity or "").lower() == "specific" and must_match_terms:
        logger.info(
            "search_semantic_scholar.specific_query",
            must_match_terms=must_match_terms[:3],
            effective_query=effective_query[:120],
        )
    logger.info("search_semantic_scholar.start", query_preview=effective_query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(effective_query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_semantic_scholar_impl(effective_query, cache_key),
    )

    logger.info("search_semantic_scholar.complete", result_count=len(result))
    return result
