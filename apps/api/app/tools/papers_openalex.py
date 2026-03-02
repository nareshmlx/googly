"""OpenAlex paper fetching tool.

All public functions return list[dict] on success, [] on any failure — never raise.
The agent layer must never see exceptions from tools.

OpenAlex polite pool: providing a mailto param routes requests through a faster,
dedicated pool. Set OPENALEX_EMAIL to any valid address to opt in.

Integrated with production infrastructure: rate limiting, circuit breaker, caching, retry.
"""

import hashlib
import json
import re
from datetime import UTC, datetime

import httpx
import structlog

from app.core.cache import TwoTierCache
from app.core.circuit_breaker import call_with_circuit_breaker, openalex_breaker
from app.core.config import settings
from app.core.dedup import deduplicate_request
from app.core.rate_limiter import openalex_limiter, rate_limited_call
from app.core.redis import get_redis
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)

_OPENALEX_BASE_URL = "https://api.openalex.org/works"
_TIMEOUT = 10.0  # seconds
_TARGET_PAPER_COUNT = 20
_OPENALEX_VARIANT_LIMIT = 3
_OPENALEX_PER_PAGE = 50
_STRICT_PAGES: tuple[int, ...] = (1, 2)

_LATEST_QUERY_TERMS: set[str] = {
    "latest",
    "recent",
    "newest",
    "current",
    "today",
}

_QUERY_STOPWORDS: set[str] = {
    "get",
    "give",
    "show",
    "find",
    "me",
    "the",
    "a",
    "an",
    "latest",
    "recent",
    "newest",
    "current",
    "research",
    "paper",
    "papers",
    "study",
    "studies",
    "on",
    "about",
    "for",
    "please",
    "and",
    "or",
    "with",
    "without",
    "from",
    "into",
    "across",
    "between",
}

_GENERIC_QUERY_TERMS: set[str] = {
    "latest",
    "recent",
    "newest",
    "current",
    "research",
    "paper",
    "papers",
    "study",
    "studies",
}

_ALLOWED_WORK_TYPES: set[str] = {
    "article",
    "review",
    "preprint",
    "proceedings-article",
    "book-chapter",
}

_PROMO_TITLE_PATTERNS: tuple[str, ...] = (
    r"\bdiscount\b",
    r"\breferral\b",
    r"\bcoupon\b",
    r"\bpromo\b",
    r"\bcashback\b",
    r"\bvoucher\b",
    r"\bcode\b",
    r"\bsave up to\b",
    r"\border\b",
)


def _tokenize(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens for lightweight ranking."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def _tokenize_ordered(text: str) -> list[str]:
    """Return unique lowercase tokens in original order (stable for query building)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _looks_like_latest_query(query: str) -> bool:
    """Detect if user is explicitly asking for newest/recent papers."""
    return bool(_tokenize(query) & _LATEST_QUERY_TERMS)


def _normalize_title(title: str) -> str:
    """Normalize paper title for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _build_query_variants(query: str) -> list[str]:
    """Build a minimal set of query variants for OpenAlex retrieval."""
    base = re.sub(r"\s+", " ", (query or "").strip())
    if not base:
        return []

    query_tokens_ordered = _tokenize_ordered(base)
    query_tokens = set(query_tokens_ordered)
    compact_tokens = [t for t in query_tokens_ordered if t not in _QUERY_STOPWORDS]
    compact = " ".join(compact_tokens[:8]).strip()
    primary = compact or base
    variants: list[str] = [primary]

    expanded_terms = list(compact_tokens)
    if "skin" in query_tokens and "skincare" not in expanded_terms:
        expanded_terms.append("skincare")
    if "hair" in query_tokens and "haircare" not in expanded_terms:
        expanded_terms.append("haircare")
    if "beauty" in query_tokens and "cosmetics" not in expanded_terms:
        expanded_terms.append("cosmetics")
    expanded_variant = " ".join(expanded_terms[:10]).strip()
    if expanded_variant:
        variants.append(expanded_variant)

    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        cleaned = re.sub(r"[,;:]+", " ", v).strip()
        key = cleaned.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out[:_OPENALEX_VARIANT_LIMIT]


def _paper_relevance_score(
    paper: dict, query_tokens: set[str], specific_query_terms: set[str]
) -> tuple[int, int, int, int]:
    """Compute lexical relevance score and overlap signals for ranking."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0, 0, 0, 0
    query_overlap = len(text_tokens & query_tokens)
    specific_overlap = len(text_tokens & specific_query_terms) if specific_query_terms else 0
    phrase_bonus = 0
    if len(specific_query_terms) >= 2 and specific_overlap >= 2:
        phrase_bonus = 2
    score = (query_overlap * 3) + (specific_overlap * 4) + phrase_bonus
    return score, query_overlap, specific_overlap, phrase_bonus


def _is_promotional_title(title: str) -> bool:
    """Return True for obvious non-research promotional/coupon style titles."""
    t = (title or "").lower()
    return any(re.search(pattern, t) for pattern in _PROMO_TITLE_PATTERNS)


def _specific_query_terms(query_tokens: set[str]) -> set[str]:
    """Return query terms that carry specific topical intent beyond generic wording."""
    generic = _QUERY_STOPWORDS | _GENERIC_QUERY_TERMS
    return {t for t in query_tokens if t not in generic}


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """
    Reconstruct a plain-text abstract from OpenAlex's abstract_inverted_index format.

    OpenAlex stores abstracts as an inverted index mapping each word to the list of
    positions it occupies in the original text. Sorting by position and joining
    recovers the original word order.

    Returns an empty string when the index is absent or empty.
    """
    if not inverted_index:
        return ""

    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    word_positions.sort()
    return " ".join(word for _, word in word_positions)


def _cache_key(project_id: str, query: str) -> str:
    """Generate deterministic project-scoped cache key for a query."""
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"search:cache:{project_id}:openalex:papers:{query_hash}"


async def _get_works_with_retry(params: dict) -> dict | None:
    """
    Internal GET helper for the OpenAlex /works endpoint with retry logic.

    Uses production infrastructure: retry with exponential backoff, rate limiting.
    Returns parsed JSON dict on success, None on any failure (per AGENTS.md Rule 4).
    """

    async def _fetch() -> dict:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(_OPENALEX_BASE_URL, params=params)
            response.raise_for_status()
            return response.json()

    # Wrap in rate limiter
    result = await retry_with_backoff(
        lambda: rate_limited_call(
            openalex_limiter,
            "openalex",
            _fetch,
        ),
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
    )
    return result


async def _fetch_papers_openalex_impl(project_id: str, query: str, cache_key: str) -> list[dict]:
    """
    Internal implementation (called by dedup layer).
    Handles cache check, API call, and cache write with stale fallback.
    """
    redis_client = await get_redis()
    cache = TwoTierCache(redis_client)

    # 1. Primary Cache Check (Fresh result)
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached.get("papers", [])

    # 2. Fetch from API
    async def _fetch_impl() -> list[dict]:
        return await _fetch_papers_impl(query)

    result = await call_with_circuit_breaker(openalex_breaker, _fetch_impl)

    # 3. Success -> Cache and Return
    if result:
        await cache.set(cache_key, {"papers": result}, ttl=settings.CACHE_TTL_PAPERS)
        # Store a "stale" copy that lasts longer
        stale_key = f"{cache_key}:stale"
        await redis_client.setex(
            stale_key, settings.CACHE_TTL_STALE, json.dumps({"papers": result})
        )
        return result

    # 4. Failure -> Serve Stale Fallback
    stale_key = f"{cache_key}:stale"
    stale_raw = await redis_client.get(stale_key)
    if stale_raw and isinstance(stale_raw, str | bytes):
        logger.warning(
            "papers_openalex.api_failed.serving_stale",
            project_id=project_id,
            query=query[:50],
        )
        try:
            return json.loads(stale_raw).get("papers", [])
        except (json.JSONDecodeError, TypeError):
            pass

    return []


async def fetch_papers(project_id: str, query: str) -> list[dict]:
    """
    Fetch research papers from OpenAlex matching the given query string.

    Integrated with production infrastructure:
    - Request deduplication (prevents thundering herd on external APIs)
    - Two-tier caching (L1 in-memory + L2 Redis) for query results
    - Circuit breaker protection for API failures
    - Rate limiting (1 req/sec for Semantic Scholar tier)
    - Retry with exponential backoff

    Calls the OpenAlex /works search endpoint, returning the top 10 results.
    Each result is normalised into a flat dict with consistent keys so the
    agent layer can treat all source types uniformly.

    The mailto param routes the request through OpenAlex's polite pool for
    better reliability — set OPENALEX_EMAIL in config to opt in.

    Returned keys per paper:
      - paper_id (str):          OpenAlex work URL, e.g. "https://openalex.org/W..."
      - title (str):             display_name from OpenAlex
      - abstract (str):          reconstructed from abstract_inverted_index; "" if absent
      - publication_year (int):  year published; 0 if missing
      - doi (str):               DOI string; "" if None
      - source (str):            always "paper"

    Returns [] on timeout, HTTP error, empty results, or any unexpected failure.
    Never raises.
    """
    logger.info("papers_openalex.fetch.start", query_preview=query[:80])

    # Generate cache key (instrumented with project_id for multi-tenant security)
    cache_key = _cache_key(project_id, query)

    # Deduplicate concurrent identical requests
    result = await deduplicate_request(
        cache_key,
        lambda: _fetch_papers_openalex_impl(project_id, query, cache_key),
    )

    logger.info("papers_openalex.fetch.success", paper_count=len(result))
    return result


async def _fetch_papers_impl(query: str) -> list[dict]:
    """
    Internal implementation of paper fetching (without cache/circuit breaker).

    Builds query variants, fetches from OpenAlex, deduplicates, ranks, and returns top 10.
    """

    query_variants = _build_query_variants(query)
    if not query_variants:
        return []

    all_works: list[dict] = []
    variant_counts: list[dict[str, int | str]] = []
    latest_like = _looks_like_latest_query(query)
    from_year = max(2018, datetime.now(UTC).year - 7)
    for variant in query_variants:
        variant_total = 0
        for page in _STRICT_PAGES:
            filter_parts = [
                f"title_and_abstract.search:{variant}",
                "has_abstract:true",
                "type:article|review|preprint|proceedings-article",
                "language:en",
            ]
            if latest_like:
                filter_parts.append(f"from_publication_date:{from_year}-01-01")
            params: dict = {
                "filter": ",".join(filter_parts),
                "per-page": _OPENALEX_PER_PAGE,
                "page": page,
                "mailto": settings.OPENALEX_EMAIL,
                "sort": "publication_date:desc",
            }
            try:
                data = await _get_works_with_retry(params)
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "papers_openalex.http_error",
                    status_code=exc.response.status_code,
                    params=params,
                )
                continue
            except Exception as exc:
                try:
                    logger.exception("papers_openalex.unexpected_error", params=params)
                except Exception:
                    logger.error(
                        "papers_openalex.unexpected_error_fallback",
                        error_type=type(exc).__name__,
                    )
                continue

            if data is None:
                continue
            results = data.get("results")
            if not results or not isinstance(results, list):
                continue
            all_works.extend(results)
            variant_total += len(results)
        variant_counts.append({"query": variant[:80], "count": variant_total})

    # Broaden only when strict title+abstract search under-fetches.
    if len(all_works) < (_TARGET_PAPER_COUNT * 3):
        for variant in query_variants:
            filter_parts = [
                "type:article|review|preprint|proceedings-article",
                "language:en",
            ]
            if latest_like:
                filter_parts.append(f"from_publication_date:{from_year}-01-01")
            params = {
                "search": variant,
                "filter": ",".join(filter_parts),
                "per-page": _OPENALEX_PER_PAGE,
                "page": 1,
                "mailto": settings.OPENALEX_EMAIL,
                "sort": "publication_date:desc",
            }
            try:
                data = await _get_works_with_retry(params)
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "papers_openalex.http_error",
                    status_code=exc.response.status_code,
                    params=params,
                )
                continue
            except Exception as exc:
                try:
                    logger.exception("papers_openalex.unexpected_error", params=params)
                except Exception:
                    logger.error(
                        "papers_openalex.unexpected_error_fallback",
                        error_type=type(exc).__name__,
                    )
                continue

            if data is None:
                continue
            results = data.get("results")
            if not results or not isinstance(results, list):
                continue
            all_works.extend(results)
            variant_counts.append({"query": f"{variant[:70]} [broad]", "count": len(results)})

    if not all_works:
        logger.info("papers_openalex.fetch.empty", query_preview=query[:80])
        return []

    papers: list[dict] = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    drop_reason_counts: dict[str, int] = {}
    for work in all_works:
        try:
            title = work.get("display_name") or ""
            doi = work.get("doi") or ""
            title_key = _normalize_title(title)
            doi_key = str(doi).strip().lower()
            if doi_key and doi_key in seen_dois:
                continue
            if title_key and title_key in seen_titles:
                continue

            if _is_promotional_title(title):
                drop_reason_counts["promo_title"] = drop_reason_counts.get("promo_title", 0) + 1
                continue

            work_type = str(work.get("type") or "").strip().lower()
            cited_by_count = int(work.get("cited_by_count") or 0)
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            abstract_len = len(abstract.strip())
            if (
                work_type
                and work_type not in _ALLOWED_WORK_TYPES
                and cited_by_count <= 0
                and abstract_len < 80
            ):
                drop_reason_counts["low_quality_type"] = (
                    drop_reason_counts.get("low_quality_type", 0) + 1
                )
                continue

            if doi_key:
                seen_dois.add(doi_key)
            if title_key:
                seen_titles.add(title_key)

            primary_location = work.get("primary_location") or {}
            source = (
                (primary_location.get("source") or {}) if isinstance(primary_location, dict) else {}
            )
            best_oa_location = work.get("best_oa_location") or {}
            open_access = work.get("open_access") or {}
            paper: dict = {
                "paper_id": work.get("id", ""),
                "title": title,
                "abstract": abstract,
                "publication_year": work.get("publication_year") or 0,
                "doi": doi,
                "url": primary_location.get("landing_page_url")
                if isinstance(primary_location, dict)
                else "",
                "pdf_url": primary_location.get("pdf_url")
                if isinstance(primary_location, dict)
                else "",
                "open_access_url": (
                    best_oa_location.get("pdf_url") if isinstance(best_oa_location, dict) else ""
                ),
                "is_open_access": bool(open_access.get("is_oa"))
                if isinstance(open_access, dict)
                else False,
                "cited_by_count": cited_by_count,
                "type": work_type,
                "source_name": source.get("display_name") if isinstance(source, dict) else "",
                "source": "paper",
            }
            papers.append(paper)
        except Exception:
            logger.warning(
                "papers_openalex.fetch.skip_work",
                work_id=work.get("id"),
                reason="failed to extract fields",
            )
            continue

    query_tokens = _tokenize(query)
    specific_terms = _specific_query_terms(query_tokens)
    ranked: list[tuple[int, int, int, int, dict]] = []
    for paper in papers:
        score, query_overlap, specific_overlap, phrase_bonus = _paper_relevance_score(
            paper,
            query_tokens,
            specific_terms,
        )
        ranked.append((score, query_overlap, specific_overlap, phrase_bonus, paper))

    pre_positive_count = len([item for item in ranked if item[0] > 0])
    strict_ranked = [item for item in ranked if item[0] > 0 and (not specific_terms or item[2] > 0)]
    positive_ranked = [item for item in ranked if item[0] > 0]
    if len(strict_ranked) >= _TARGET_PAPER_COUNT:
        ranked_pool = strict_ranked
    elif len(positive_ranked) >= _TARGET_PAPER_COUNT:
        ranked_pool = positive_ranked
    else:
        ranked_pool = ranked

    ranked_pool.sort(
        key=lambda item: (
            item[0],
            item[2],
            item[1],
            int(item[4].get("publication_year") or 0),
            int(item[4].get("cited_by_count") or 0),
        ),
        reverse=True,
    )

    selected = (
        [paper for _, _, _, _, paper in ranked_pool[:_TARGET_PAPER_COUNT]]
        if ranked_pool
        else papers[:_TARGET_PAPER_COUNT]
    )

    filtered_out = [item for item in ranked if item not in ranked_pool][:10]
    filtered_out_samples = [
        {
            "title": str(item[4].get("title") or "")[:90],
            "year": int(item[4].get("publication_year") or 0),
            "score": int(item[0]),
            "query_overlap": int(item[1]),
            "specific_overlap": int(item[2]),
            "phrase_bonus": int(item[3]),
        }
        for item in filtered_out[:5]
    ]
    logger.info(
        "papers_openalex.filtering_summary",
        query_preview=query[:80],
        variant_counts=variant_counts,
        latest_like=latest_like,
        specific_terms=sorted(specific_terms)[:8],
        raw_count=len(all_works),
        deduped_count=len(papers),
        positive_scored=pre_positive_count,
        selected_count=len(selected),
        drop_reason_counts=drop_reason_counts,
        filtered_out_samples=filtered_out_samples,
    )
    logger.info("papers_openalex.fetch.success", paper_count=len(selected))
    return selected
