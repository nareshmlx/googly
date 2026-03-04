"""Perigon news search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

Perigon provides news aggregation with source filtering and relevance ranking.
Plus plan: 4 req/sec, $0.011 per request (50k req/mo = $550).

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import time
from datetime import UTC, datetime, timedelta

import httpx
import structlog

from app.core.cache_keys import build_search_cache_key
from app.core.circuit_breaker import call_with_circuit_breaker, perigon_breaker
from app.core.config import settings
from app.core.dedup import deduplicate_request
from app.core.external_cache import cached_external_results
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.query_sanitize import sanitize_query
from app.core.rate_limiter import perigon_limiter, rate_limited_call
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_PERIGON_BASE_URL = "https://api.goperigon.com/v1/all"
_CATEGORIES = "Business,Tech,Lifestyle"


def _build_perigon_query(
    query: str,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
) -> str:
    """Compose Perigon query preserving specific entities while adding context."""
    must_match_terms = [str(term).strip() for term in (must_match_terms or []) if str(term).strip()]
    domain_terms = [str(term).strip() for term in (domain_terms or []) if str(term).strip()]
    if must_match_terms:
        parts = [f'"{term}"' for term in must_match_terms]
        parts.extend(domain_terms)
        return " ".join(parts).strip()
    return query


def _build_perigon_categories(domain_terms: list[str] | None) -> str:
    """Choose Perigon categories dynamically for better topical precision."""
    terms = {str(term).lower().strip() for term in (domain_terms or []) if str(term).strip()}
    if terms & {"skincare", "dermatology", "cosmetics", "fragrance"}:
        return "Health,Lifestyle,Business"
    return _CATEGORIES


def _cache_key(project_id: str, query: str) -> str:
    """Generate deterministic project-scoped cache key for a query."""
    return build_search_cache_key(
        project_id=project_id,
        provider="perigon",
        query_type="news",
        parts=[query],
    )


async def _search_with_retry(query: str, category: str) -> dict | None:
    """
    Internal GET helper for the Perigon /v1/all endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).
    """
    # Calculate date range (last 30 days)
    from_date = (datetime.now(UTC) - timedelta(days=settings.PERIGON_DAYS_LOOKBACK)).isoformat()

    async def _fetch() -> dict:
        async with httpx.AsyncClient(timeout=settings.PERIGON_TIMEOUT_SECONDS) as client:
            response = await client.get(
                _PERIGON_BASE_URL,
                params={
                    "apiKey": settings.PERIGON_API_KEY,
                    "q": query,
                    "sortBy": "relevance",
                    "showReprints": "false",
                    "showNumResults": "true",
                    "category": category,
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


async def _search_impl(query: str, category: str) -> list[dict]:
    """
    Internal implementation of Perigon news search (without cache/circuit breaker).

    Fetches from Perigon API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query, category)
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
            timeout=settings.PERIGON_TIMEOUT_SECONDS,
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
        api_calls_total.labels(api_name="perigon", status="error").inc()
        logger.warning("news_perigon.no_data", query_preview=query[:80])
        return []

    api_calls_total.labels(api_name="perigon", status="success").inc()

    articles = data.get("articles", [])
    if not articles or not isinstance(articles, list):
        logger.info("news_perigon.empty_results", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for article in articles:
        try:
            # Extract source name safely — Perigon uses "domain" not "name"
            source_name = ""
            if isinstance(article.get("source"), dict):
                source_name = article["source"].get("domain", "")

            normalized.append(
                {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "content": article.get("description", ""),
                    "source": "perigon",
                    "published_date": article.get("pubDate"),
                    "source_name": source_name,
                }
            )
        except Exception:
            logger.exception("news_perigon.result_normalization_error", article=article)
            continue

    logger.info("news_perigon.success", result_count=len(normalized))
    return normalized


async def _search_perigon_impl(
    project_id: str, query: str, cache_key: str, category: str
) -> list[dict]:
    """
    Internal implementation (called by dedup layer).
    Handles cache check, API call, and cache write with stale fallback.
    """
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query, category)

    return await cached_external_results(
        cache_key=cache_key,
        fetch_fn=lambda: call_with_circuit_breaker(perigon_breaker, _fetch_impl),
        ttl=settings.CACHE_TTL_NEWS,
        stale_event="news_perigon.api_failed.serving_stale",
        stale_context={"project_id": project_id, "query": query[:50]},
    )


async def search_perigon(
    project_id: str,
    query: str,
    *,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
    query_specificity: str | None = None,
) -> list[dict]:
    """
    Search news articles using Perigon API with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (4 req/sec for Plus plan)
    - Retry with exponential backoff

    Perigon provides news aggregation with source filtering and relevance ranking.
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

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Validate API key is set
    if not settings.PERIGON_API_KEY:
        logger.error("news_perigon.api_key_missing")
        return []

    # Sanitize query
    sanitized_query = sanitize_query(
        query,
        logger=logger,
        empty_event="news_perigon.empty_query",
        too_long_event="news_perigon.query_too_long",
    )
    if sanitized_query is None:
        return []
    query = sanitized_query

    effective_query = _build_perigon_query(query, must_match_terms, domain_terms)
    category = _build_perigon_categories(domain_terms)
    if str(query_specificity or "").lower() == "specific" and must_match_terms:
        logger.info(
            "news_perigon.specific_query",
            must_match_terms=must_match_terms,
            category=category,
        )

    logger.info("news_perigon.start", query_preview=effective_query[:80], category=category)

    # Generate cache key (instrumented with project_id for multi-tenant security)
    cache_key = _cache_key(project_id, f"{effective_query}|{category}")

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_perigon_impl(project_id, effective_query, cache_key, category),
    )

    logger.info("news_perigon.complete", result_count=len(result))
    return result
