"""PubMed research paper search tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

PubMed provides access to 35+ million biomedical citations from MEDLINE, life science journals, and online books.
Rate limit: 10 req/sec (with API key), 3 req/sec (without API key).

Two-step API process:
1. Search for PMIDs (PubMed IDs) using esearch.fcgi
2. Fetch paper details using esummary.fcgi

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import time

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, pubmed_breaker
from app.core.config import CacheKeys, settings
from app.core.dedup import deduplicate_request
from app.core.metrics import api_call_duration_seconds, api_calls_total
from app.core.rate_limiter import pubmed_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_TIMEOUT = 15.0  # seconds (per request, 2 requests total)
_MAX_RESULTS = 20

# Module-level HTTP client pool (reused across requests to prevent memory leak)
_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client pool for module."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _http_client


def _build_pubmed_query(
    query: str,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
) -> str:
    """Build PubMed query with Title/Abstract constraints for specific terms."""
    must_match_terms = [str(term).strip() for term in (must_match_terms or []) if str(term).strip()]
    domain_terms = [str(term).strip() for term in (domain_terms or []) if str(term).strip()]
    if not must_match_terms:
        return query
    must_expr = " OR ".join(f'"{term}"[Title/Abstract]' for term in must_match_terms[:3])
    must_clause = f"({must_expr})"
    if not domain_terms:
        return must_clause
    domain_expr = " OR ".join(f'"{term}"[Title/Abstract]' for term in domain_terms[:2])
    return f"{must_clause} AND ({domain_expr})"


def _best_available_content(paper: dict, title: str, authors: list[str]) -> str:
    """Build content from best available summary fields when abstract is unavailable."""
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        return abstract

    candidate_values = [
        str(paper.get("elocationid") or "").strip(),
        str(paper.get("booktitle") or "").strip(),
        str(paper.get("source") or "").strip(),
        str(paper.get("pubdate") or "").strip(),
        ", ".join(authors[:3]).strip() if authors else "",
    ]
    title_key = title.strip().lower()

    parts: list[str] = []
    seen: set[str] = set()
    for value in candidate_values:
        key = value.lower()
        if not value or key == title_key or key in seen:
            continue
        seen.add(key)
        parts.append(value)

    if parts:
        return " | ".join(parts)
    return title


def _cache_key(query: str) -> str:
    """Generate deterministic cache key for a query."""
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"{CacheKeys.PAPERS_PUBMED}:{query_hash}"


async def _fetch_ids_internal(query: str) -> list[str]:
    """
    Internal helper: Fetch PubMed IDs (PMIDs) matching the query.

    Uses esearch.fcgi endpoint. No rate limiting (caller handles it).
    Returns list of PMIDs on success.
    Raises httpx exceptions on failure (caller handles error logging).
    """
    params: dict[str, str | int] = {
        "db": "pubmed",
        "term": query,
        "retmax": _MAX_RESULTS,
        "retmode": "json",
    }
    # Add API key if available
    if hasattr(settings, "PUBMED_API_KEY") and settings.PUBMED_API_KEY:
        params["api_key"] = settings.PUBMED_API_KEY

    client = await _get_http_client()
    response = await client.get(_SEARCH_URL, params=params)
    response.raise_for_status()
    data = response.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    return id_list if isinstance(id_list, list) else []


async def _fetch_details_internal(pmids: list[str]) -> dict:
    """
    Internal helper: Fetch paper details for a list of PubMed IDs.

    Uses esummary.fcgi endpoint. No rate limiting (caller handles it).
    Returns parsed JSON dict on success.
    Raises httpx exceptions on failure (caller handles error logging).
    """
    params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    # Add API key if available
    if hasattr(settings, "PUBMED_API_KEY") and settings.PUBMED_API_KEY:
        params["api_key"] = settings.PUBMED_API_KEY

    client = await _get_http_client()
    response = await client.get(_SUMMARY_URL, params=params)
    response.raise_for_status()
    return response.json()


async def _search_with_retry(query: str) -> dict | None:
    """
    Internal two-step search helper: fetch IDs → fetch details.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    CRITICAL: Rate limiting happens ONCE for the entire operation (not per API call).
    This prevents double rate limiting that was making PubMed 2× slower than needed.

    Returns parsed JSON dict with paper details on success, None on any failure.
    """

    async def _fetch_both() -> dict:
        # Step 1: Get PMIDs (no inner rate limiting)
        pmids = await _fetch_ids_internal(query)
        if not pmids:
            return {}

        # Step 2: Get paper details (no inner rate limiting)
        details = await _fetch_details_internal(pmids)
        return details

    # Wrap ENTIRE operation in rate limiter + retry (rate limit once, not twice)
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            pubmed_limiter,
            "pubmed",
            _fetch_both,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _search_impl(query: str) -> list[dict]:
    """
    Internal implementation of PubMed search (without cache/circuit breaker).

    Fetches from PubMed API and normalizes results into consistent dict format.
    """
    start_time = time.perf_counter()
    try:
        data = await _search_with_retry(query)
        api_calls_total.labels(api_name="pubmed", status="success").inc()
    except httpx.HTTPStatusError as exc:
        api_calls_total.labels(api_name="pubmed", status="error").inc()
        logger.error(
            "search_pubmed.http_error",
            status_code=exc.response.status_code,
            response_preview=exc.response.text[:200] if exc.response.text else "",
            query_preview=query[:80],
        )
        return []
    except httpx.TimeoutException:
        api_calls_total.labels(api_name="pubmed", status="timeout").inc()
        logger.warning(
            "search_pubmed.timeout",
            timeout=_TIMEOUT,
            query_preview=query[:80],
        )
        return []
    except Exception:
        api_calls_total.labels(api_name="pubmed", status="error").inc()
        logger.exception("search_pubmed.unexpected_error", query_preview=query[:80])
        return []
    finally:
        duration = time.perf_counter() - start_time
        api_call_duration_seconds.labels(api_name="pubmed").observe(duration)

    if data is None:
        logger.warning("search_pubmed.no_data", query_preview=query[:80])
        return []

    # PubMed returns {"result": {"uids": [...], "<pmid>": {...}, ...}}
    result = data.get("result", {})
    if not result or not isinstance(result, dict):
        logger.info("search_pubmed.empty_results", query_preview=query[:80])
        return []

    # Extract UIDs (PMIDs) from result
    uids = result.get("uids", [])
    if not uids or not isinstance(uids, list):
        logger.info("search_pubmed.no_uids", query_preview=query[:80])
        return []

    # Normalize results into consistent format
    normalized = []
    for pmid in uids:
        try:
            paper = result.get(str(pmid))
            if not paper or not isinstance(paper, dict):
                continue

            # Extract authors (array of dicts with "name" key)
            authors_raw = paper.get("authors", [])
            authors = []
            if isinstance(authors_raw, list):
                for author in authors_raw:
                    if isinstance(author, dict) and "name" in author:
                        authors.append(author["name"])

            title = str(paper.get("title") or "")
            content = _best_available_content(paper, title, authors)

            normalized.append(
                {
                    "title": title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "content": content,
                    "authors": authors,
                    "published": paper.get("pubdate", ""),
                    "pmid": str(pmid),
                    "source": "pubmed",
                    "journal": paper.get("source", ""),
                }
            )
        except Exception:
            logger.exception("search_pubmed.result_normalization_error", pmid=pmid)
            continue

    logger.info("search_pubmed.success", result_count=len(normalized))
    return normalized


async def _search_pubmed_impl(query: str, cache_key: str) -> list[dict]:
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
        logger.info("search_pubmed.cache.hit", result_count=len(cached.get("results", [])))
        return cached.get("results", [])

    # Wrap entire search operation in circuit breaker
    async def _fetch_impl() -> list[dict]:
        return await _search_impl(query)

    result = await call_with_circuit_breaker(
        pubmed_breaker,
        _fetch_impl,
    )

    # Circuit breaker returns [] on failure, or list[dict] on success
    if result is None or not isinstance(result, list):
        result = []

    # Cache successful results (TTL from config - research papers don't change frequently)
    if result:
        await cache.set(cache_key, {"results": result}, ttl=settings.CACHE_TTL_PAPERS)

    return result


async def search_pubmed(
    query: str,
    *,
    must_match_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
    query_specificity: str | None = None,
) -> list[dict]:
    """
    Search PubMed for research papers with the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (10 req/sec with API key, 3 req/sec without)
    - Retry with exponential backoff

    PubMed provides access to 35+ million biomedical citations from MEDLINE,
    life science journals, and online books.

    Two-step API process:
    1. Search for PMIDs (PubMed IDs) using esearch.fcgi
    2. Fetch paper details using esummary.fcgi

    Returned keys per result:
      - title (str):       Paper title
      - url (str):         PubMed URL (https://pubmed.ncbi.nlm.nih.gov/{pmid}/)
      - content (str):     Paper title (abstracts require additional API call)
      - authors (list):    List of author names
      - published (str):   Publication date
      - pmid (str):        PubMed ID
      - source (str):      Always "pubmed"
      - journal (str):     Journal name

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    # Sanitize and normalize query BEFORE cache key generation (prevents cache key collision)
    query = query.strip().lower()
    if not query:
        logger.warning("search_pubmed.empty_query")
        return []
    if len(query) > 1000:
        logger.warning("search_pubmed.query_too_long", original_length=len(query))
        query = query[:1000]

    effective_query = _build_pubmed_query(query, must_match_terms, domain_terms)
    if str(query_specificity or "").lower() == "specific" and must_match_terms:
        logger.info(
            "search_pubmed.specific_query",
            must_match_terms=must_match_terms[:3],
            effective_query=effective_query[:120],
        )
    logger.info("search_pubmed.start", query_preview=effective_query[:80])

    # Generate cache key (used for both dedup and cache)
    cache_key = _cache_key(effective_query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _search_pubmed_impl(effective_query, cache_key),
    )

    logger.info("search_pubmed.complete", result_count=len(result))
    return result
