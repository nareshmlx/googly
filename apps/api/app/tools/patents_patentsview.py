"""PatentsView API tool for patent search.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

PatentsView provides access to USPTO patent data via their v2 Search API
(search.patentsview.org). An API key is required — register at
https://patentsview.org/apis/api-faqs. Rate limit: 45 req/min.

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.

NOTE: The legacy API (api.patentsview.org) was shut down on May 1, 2025.
      This tool targets the new Search API endpoint.
"""

import json
import time

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.cache_keys import build_search_cache_key, build_stale_cache_key
from app.core.circuit_breaker import call_with_circuit_breaker, patentsview_breaker
from app.core.config import settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.query_sanitize import sanitize_query
from app.core.rate_limiter import patentsview_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_BASE_URL = "https://search.patentsview.org/api/v1/patent/"

# Module-level HTTP client pool (reused across requests to prevent memory leak)
_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client pool for module."""
    global _http_client
    if _http_client is None:
        headers = {}
        if settings.PATENTSVIEW_API_KEY:
            headers["X-Api-Key"] = settings.PATENTSVIEW_API_KEY
        _http_client = httpx.AsyncClient(
            timeout=settings.PATENTS_PATENTSVIEW_TIMEOUT_SECONDS, headers=headers
        )
    return _http_client


def _cache_key(project_id: str, query: str) -> str:
    """Generate deterministic project-scoped cache key for a query."""
    return build_search_cache_key(
        project_id=project_id,
        provider="patentsview",
        query_type="patents",
        parts=[query],
    )


def _build_patentsview_query(
    query: str,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
) -> dict:
    """Build a recall-oriented PatentsView query across title and abstract fields."""
    candidate_terms = [
        str(term).strip()
        for term in [query, *(must_match_terms or []), *(domain_terms or [])]
        if str(term).strip()
    ]
    unique_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in candidate_terms:
        normalized = term.lower()
        if normalized in seen_terms:
            continue
        seen_terms.add(normalized)
        unique_terms.append(term)

    clauses = [
        {"_text_all": {"patent_title": term}}
        for term in unique_terms
    ] + [
        {"_text_all": {"patent_abstract": term}}
        for term in unique_terms
    ]
    if not clauses:
        return {"_text_all": {"patent_title": query}}
    if len(clauses) == 1:
        return clauses[0]
    return {"_or": clauses}


async def _search_with_retry(query_payload: dict) -> dict | None:
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
                "q": query_payload,
                "f": [
                    "patent_id",
                    "patent_title",
                    "patent_abstract",
                    "patent_date",
                    "inventors.inventor_name_first",
                    "inventors.inventor_name_last",
                ],
                "o": {"size": settings.PAPERS_MAX_RESULTS},
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


async def _search_impl(query: str, query_payload: dict) -> list[dict]:
    """
    Internal implementation of PatentsView search (without cache/circuit breaker).

    Fetches from PatentsView API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query_payload)
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
            timeout=settings.PATENTS_PATENTSVIEW_TIMEOUT_SECONDS,
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
            # New API uses patent_id (was patent_number in legacy API)
            patent_id = p.get("patent_id", "")
            google_patents_url = (
                f"https://patents.google.com/patent/US{patent_id}" if patent_id else ""
            )

            # New API nests inventors: [{"inventor_name_first": "...", "inventor_name_last": "..."}]
            inventors = []
            inventor_list = p.get("inventors", [])
            if not isinstance(inventor_list, list):
                inventor_list = []

            for inv in inventor_list:
                if not isinstance(inv, dict):
                    continue
                first_str = inv.get("inventor_name_first") or ""
                last_str = inv.get("inventor_name_last") or ""
                if first_str or last_str:
                    inventors.append({"first_name": first_str, "last_name": last_str})

            normalized.append(
                {
                    "title": p.get("patent_title", ""),
                    "url": google_patents_url,
                    "content": p.get("patent_abstract", ""),
                    "patent_number": patent_id,
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


async def _search_patentsview_impl(
    project_id: str, query: str, cache_key: str, query_payload: dict
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
        return await _search_impl(query, query_payload)

    result = await call_with_circuit_breaker(patentsview_breaker, _fetch_impl)

    # 3. Success -> Cache and Return
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_PATENTS)
        # Store a "stale" copy that lasts longer
        stale_key = build_stale_cache_key(cache_key)
        await redis_client.setex(
            stale_key, settings.CACHE_TTL_STALE, json.dumps({"results": result})
        )
        return result

    # 4. Failure -> Serve Stale Fallback
    stale_key = build_stale_cache_key(cache_key)
    stale_raw = await redis_client.get(stale_key)
    if stale_raw:
        logger.warning(
            "patents_patentsview.api_failed.serving_stale",
            project_id=project_id,
            query=query[:50],
        )
        try:
            return json.loads(stale_raw).get("results", [])
        except json.JSONDecodeError:
            pass

    return []


async def search_patentsview(
    project_id: str,
    query: str,
    *,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
    query_specificity: str | None = None,
) -> list[dict]:
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
    sanitized_query = sanitize_query(
        query,
        logger=logger,
        empty_event="patents_patentsview.empty_query",
        too_long_event="patents_patentsview.query_too_long",
        lower=True,
    )
    if sanitized_query is None:
        return []
    query = sanitized_query

    query_payload = _build_patentsview_query(query, must_match_terms, domain_terms)
    if str(query_specificity or "").lower() == "specific" and must_match_terms:
        logger.info(
            "patents_patentsview.specific_query",
            must_match_terms=must_match_terms,
        )
    logger.info("patents_patentsview.start", query_preview=query[:80])

    # Generate cache key (instrumented with project_id for multi-tenant security)
    cache_key = _cache_key(project_id, f"{query}|{query_payload}")

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_patentsview_impl(project_id, query, cache_key, query_payload),
    )

    logger.info("patents_patentsview.complete", result_count=len(result))
    return result
