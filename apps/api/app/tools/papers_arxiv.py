"""arXiv academic paper search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

arXiv provides access to 2M+ preprints in physics, mathematics, CS, and more.
Free tier: 0.33 req/sec (1 request per 3 seconds), no API key required.

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time

import httpx
import structlog
from defusedxml import ElementTree

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import arxiv_breaker, call_with_circuit_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import arxiv_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_BASE_URL = "https://export.arxiv.org/api/query"
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
    return f"{CacheKeys.PAPERS_ARXIV}:{query_hash}"


async def _search_with_retry(query: str) -> str | None:
    """
    Internal GET helper for the arXiv API query endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns raw XML string on success, None on any failure (per AGENTS.md Rule 4).
    """

    async def _fetch() -> str:
        client = await _get_http_client()
        response = await client.get(
            _BASE_URL,
            params={
                "search_query": f"all:{query}",
                "max_results": _MAX_RESULTS,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        return response.text

    # Wrap in rate limiter + retry
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            arxiv_limiter,
            "arxiv",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


def _extract_arxiv_id(entry_id: str) -> str:
    """Extract arXiv ID from entry.id URL (e.g., 'http://arxiv.org/abs/2401.12345v1' -> '2401.12345')."""
    try:
        # entry.id format: http://arxiv.org/abs/2401.12345v1
        parts = entry_id.rstrip("/").split("/")
        arxiv_id_with_version = parts[-1]  # '2401.12345v1'
        # Remove version suffix (v1, v2, etc.)
        arxiv_id = (
            arxiv_id_with_version.split("v")[0]
            if "v" in arxiv_id_with_version
            else arxiv_id_with_version
        )
        return arxiv_id
    except Exception:
        return ""


def _parse_atom_feed(xml_text: str) -> list[dict]:
    """Parse Atom XML feed from arXiv API into normalized result format.

    Handles XML parsing errors gracefully, returns [] on any failure.
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        logger.exception("search_arxiv.xml_parse_error")
        return []

    # Atom namespace
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    entries = root.findall("atom:entry", ns)
    if not entries:
        return []

    normalized = []
    for entry in entries:
        try:
            # Extract fields from XML
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

            id_elem = entry.find("atom:id", ns)
            entry_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

            summary_elem = entry.find("atom:summary", ns)
            summary = (
                summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
            )

            published_elem = entry.find("atom:published", ns)
            published = (
                published_elem.text.strip()
                if published_elem is not None and published_elem.text
                else ""
            )

            # Extract authors
            author_elems = entry.findall("atom:author", ns)
            authors = []
            for author_elem in author_elems:
                name_elem = author_elem.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Extract categories
            category_elems = entry.findall("atom:category", ns)
            categories = []
            for cat_elem in category_elems:
                term = cat_elem.get("term")
                if term:
                    categories.append(term)

            # Extract arXiv ID
            arxiv_id = _extract_arxiv_id(entry_id)

            normalized.append(
                {
                    "title": title,
                    "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry_id,
                    "content": summary,
                    "authors": authors,
                    "published": published,
                    "arxiv_id": arxiv_id,
                    "source": "arxiv",
                    "categories": categories,
                }
            )
        except Exception:
            logger.exception(
                "search_arxiv.entry_parse_error",
                entry=ElementTree.tostring(entry, encoding="unicode"),
            )
            continue

    return normalized


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of arXiv search (without cache/circuit breaker).

    Fetches from arXiv API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        xml_text = await _search_with_retry(query)
        api_calls_total.labels(api_name="arxiv", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="arxiv", status="error").inc()
        logger.error(
            "search_arxiv.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="arxiv", status="timeout").inc()
        logger.warning(
            "search_arxiv.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except Exception:
        api_calls_total.labels(api_name="arxiv", status="error").inc()
        logger.exception("search_arxiv.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="arxiv").observe(duration)

    if xml_text is None:
        logger.warning("search_arxiv.no_data", query_preview=query[:80])
        return []

    # Parse XML into normalized results
    normalized = _parse_atom_feed(xml_text)

    if not normalized:
        logger.info("search_arxiv.empty_results", query_preview=query[:80])
        return []

    logger.info("search_arxiv.success", result_count=len(normalized))
    return normalized


async def _search_arxiv_impl(query: str, cache_key: str) -> list[dict]:
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
        logger.info("search_arxiv.cache.hit", result_count=len(cached.get("results", [])))
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query)

    result = await call_with_circuit_breaker(
        arxiv_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config - research papers don't change frequently)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_PAPERS)

    return result


async def search_arxiv(query: str) -> list[dict]:
    """
    Search academic papers using arXiv API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results (TTL from config)
    - Circuit breaker protection for API failures
    - Rate limiting (0.33 req/sec = 1 req per 3 sec for free tier)
    - Retry with exponential backoff

    arXiv provides access to 2M+ preprints in physics, mathematics, CS, and more.
    Free tier: 0.33 req/sec (1 request per 3 seconds), no API key required.

    Returned keys per result:
      - title (str):          Paper title
      - url (str):            Paper URL on arXiv
      - content (str):        Paper abstract (used as content)
      - authors (list[str]):  List of author names
      - published (str):      Publication date (ISO format)
      - arxiv_id (str):       arXiv ID (e.g., "2401.12345")
      - source (str):         Always "arxiv"
      - categories (list[str]): arXiv categories (e.g., ["cs.AI", "cs.LG"])

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Sanitize and normalize query BEFORE cache key generation (prevents cache key collision)
    query = query.strip().lower()
    if not query:
        logger.warning("search_arxiv.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("search_arxiv.query_too_long", original_length=len(query))
        query = query[:1000]

    logger.info("search_arxiv.start", query_preview=query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_arxiv_impl(query, cache_key),
    )

    logger.info("search_arxiv.complete", result_count=len(result))
    return result
