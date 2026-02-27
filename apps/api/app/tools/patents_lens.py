"""Lens.org Patent API tool for patent search.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

Lens.org provides free access to global patent data from 100+ jurisdictions.
Free tier: 10,000 requests/month. API key required.

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, lens_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import lens_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.lens.org/patent/search"
_TIMEOUT = 20.0  # seconds (Lens API can be slower for complex queries)
_MAX_RESULTS = 20  # Lens recommends 20-100 per request

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
    return f"{CacheKeys.PATENTS_LENS}:{query_hash}"


async def _search_with_retry(query: str) -> dict | None:
    """
    Internal POST helper for the Lens.org /patent/search endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).
    """

    async def _fetch() -> dict:
        client = await _get_http_client()

        # Check if API key is configured
        if not settings.LENS_API_KEY:
            logger.error("search_lens.no_api_key", message="LENS_API_KEY not configured")
            raise ValueError("LENS_API_KEY not configured")

        # Build headers with Bearer token
        headers = {
            "Authorization": f"Bearer {settings.LENS_API_KEY}",
            "Content-Type": "application/json",
        }

        # Build query payload using Lens.org query format
        # Use bool query with should clauses to search across title, abstract, and claims
        payload = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"title": query}},
                        {"match": {"abstract.text": query}},
                        {"match": {"claim.text": query}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": _MAX_RESULTS,
            "include": [
                "lens_id",
                "jurisdiction",
                "doc_number",
                "kind",
                "date_published",
                "biblio.invention_title",
                "biblio.parties.inventors",
                "abstract",
            ],
            "sort": [{"date_published": "desc"}],
        }

        # NOTE: Removed restrictive cosmetic-only filter (was lines 100-114).
        # Previous filter: classification.cpc:A61K8* OR classification.cpc:A61Q*
        # This was limiting results to cosmetic patents only, missing pharmaceutical,
        # chemical engineering, and drug delivery patents relevant to queries like
        # "retinol stabilization encapsulation" which span multiple IPC/CPC classes.
        # The query terms themselves provide sufficient filtering relevance.

        response = await client.post(_BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    # Wrap in rate limiter + retry
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            lens_limiter,
            "lens",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of Lens.org patent search (without cache/circuit breaker).

    Fetches from Lens.org API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query)
        api_calls_total.labels(api_name="lens", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="lens", status="error").inc()
        logger.error(
            "patents_lens.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="lens", status="timeout").inc()
        logger.warning(
            "patents_lens.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except ValueError as exc:
        # Handle missing API key
        api_calls_total.labels(api_name="lens", status="error").inc()
        logger.error("patents_lens.config_error", error=str(exc))
        return []
    except Exception:
        api_calls_total.labels(api_name="lens", status="error").inc()
        logger.exception("patents_lens.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="lens").observe(duration)

    if data is None:
        logger.warning("patents_lens.no_data", query_preview=query[:80])
        return []

    patents = data.get("data", [])
    if not patents or not isinstance(patents, list):
        logger.info("patents_lens.empty_results", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for p in patents:
        try:
            # Extract lens_id
            lens_id = p.get("lens_id", "")

            # Extract jurisdiction and doc_number to build patent URL
            jurisdiction = p.get("jurisdiction", "")
            doc_number = p.get("doc_number", "")
            kind = p.get("kind", "")

            # Build Lens.org URL for the patent
            patent_url = f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else ""

            # Extract title (can be array of objects with text/lang)
            title = ""
            invention_titles = p.get("biblio", {}).get("invention_title", [])
            if (
                invention_titles
                and isinstance(invention_titles, list)
                and len(invention_titles) > 0
            ):
                title = (
                    invention_titles[0].get("text", "")
                    if isinstance(invention_titles[0], dict)
                    else ""
                )

            # Extract abstract text
            abstract = ""
            abstract_obj = p.get("abstract", {})
            if isinstance(abstract_obj, dict):
                abstract = abstract_obj.get("text", "")
            elif isinstance(abstract_obj, list) and len(abstract_obj) > 0:
                abstract = (
                    abstract_obj[0].get("text", "") if isinstance(abstract_obj[0], dict) else ""
                )

            # Extract inventors
            inventors = []
            parties = p.get("biblio", {}).get("parties", {})
            inventors_list = parties.get("inventors", [])
            if inventors_list and isinstance(inventors_list, list):
                for inventor in inventors_list:
                    if isinstance(inventor, dict):
                        # Lens format: {"extracted_name": {"value": "John Doe"}}
                        extracted_name = inventor.get("extracted_name", {})
                        if isinstance(extracted_name, dict):
                            name = extracted_name.get("value", "")
                            if name:
                                inventors.append({"name": name})

            normalized.append(
                {
                    "title": title,
                    "url": patent_url,
                    "content": abstract,
                    "lens_id": lens_id,
                    "patent_number": f"{jurisdiction}{doc_number}"
                    if jurisdiction and doc_number
                    else doc_number,
                    "jurisdiction": jurisdiction,
                    "kind": kind,
                    "date": p.get("date_published", ""),
                    "inventors": inventors,
                    "source": "lens",
                }
            )
        except Exception:
            logger.exception("patents_lens.result_normalization_error", patent=p)
            continue

    logger.info("patents_lens.success", result_count=len(normalized))
    return normalized


async def _search_lens_impl(query: str, cache_key: str) -> list[dict]:
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
        logger.info("patents_lens.cache.hit", result_count=len(cached.get("results", [])))
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query)

    result = await call_with_circuit_breaker(
        lens_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config - patents don't change frequently)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_PATENTS)

    return result


async def search_lens(query: str) -> list[dict]:
    """
    Search for patents using Lens.org API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (conservative to stay within 10k/month free tier)
    - Retry with exponential backoff

    Lens.org provides free access to global patent data from 100+ jurisdictions.
    Free tier: 10,000 requests/month. API key required.

    Request deduplication prevents thundering herd when multiple users
    search for the same patents simultaneously.

    Returned keys per result:
      - title (str):          Patent title
      - url (str):            Lens.org patent URL (https://www.lens.org/lens/patent/{lens_id})
      - content (str):        Patent abstract
      - lens_id (str):        Lens.org unique identifier
      - patent_number (str):  Patent number with jurisdiction (e.g., "US20130227762")
      - jurisdiction (str):   Country code (US, EP, CN, etc.)
      - kind (str):           Kind code (A1, B2, etc.)
      - date (str):           Patent publication date (YYYY-MM-DD)
      - inventors (list[dict]): List of inventors with name field
      - source (str):         Always "lens"

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Check if API key is configured
    if not settings.LENS_API_KEY:
        logger.error(
            "search_lens.no_api_key",
            message="LENS_API_KEY not configured in settings. Get key from https://www.lens.org/lens/user/subscriptions",
        )
        return []

    # Sanitize and normalize query BEFORE cache key generation (prevents cache key collision)
    query = query.strip().lower()
    if not query:
        logger.warning("patents_lens.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("patents_lens.query_too_long", original_length=len(query))
        query = query[:1000]

    logger.info("patents_lens.start", query_preview=query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_lens_impl(query, cache_key),
    )

    logger.info("patents_lens.complete", result_count=len(result))
    return result
