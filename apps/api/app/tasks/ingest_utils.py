"""Shared utilities for source ingestion tasks.

extracted from ingest_project.py to support modular ingestion sources.
"""

import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import structlog

from app.core.config import settings
from app.kb.ingester import RawDocument
from app.tools.search_exa import search_exa
from app.tools.search_tavily import search_tavily

logger = structlog.get_logger(__name__)

SOCIAL_SOURCES: frozenset[str] = frozenset(
    {
        "social_tiktok",
        "social_instagram",
        "social_youtube",
        "social_reddit",
        "social_x",
    }
)

# Constants for ranking and scoring
REEL_RECENCY_DAYS = 30.0

_GENERIC_RELEVANCE_TERMS: set[str] = {
    "trend",
    "trends",
    "research",
    "analysis",
    "market",
    "industry",
    "product",
    "products",
    "news",
    "viral",
}

_SOCIAL_BROAD_TERMS: set[str] = {
    "beauty",
    "skincare",
    "makeup",
    "cosmetics",
    "fragrance",
    "viral",
    "trend",
    "trends",
    "news",
}

_PROJECT_TEXT_STOPWORDS: set[str] = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "about",
    "project",
    "research",
    "analysis",
    "find",
    "highly",
    "relevant",
    "social",
    "track",
    "conversation",
    "conversations",
    "evidence",
    "quality",
    "reports",
    "guidance",
    "projects",
    "latest",
    "recent",
    "new",
    "fast",
    "fix",
    "quickcheck",
    "check",
    "test",
    "tests",
    "using",
    "based",
    "youtube",
    "reddit",
    "twitter",
    "tiktok",
    "instagram",
    "platform",
    "platforms",
}


def _as_int(value: object) -> int:
    """Safely coerce any numeric-like object to an integer."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            if not value:
                return 0
            return int(float(value))
        return int(float(value))  # type: ignore
    except (TypeError, ValueError):
        return 0


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenization for lexical counting."""
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _match_count(text: str, terms: set[str]) -> int:
    """Count distinct relevance-term matches in text."""
    if not text or not terms:
        return 0
    tokens = _tokenize(text)
    return len(tokens & terms)


def _content_quality_score(title: str, content: str) -> float:
    """Estimate content quality from richness and specificity signals."""
    merged = f"{title} {content}".strip()
    if not merged:
        return 0.0
    tokens = _tokenize(merged)
    unique_token_score = min(1.0, len(tokens) / 80.0)
    length_score = min(1.0, len(content) / 900.0)
    punctuation_count = content.count(".") + content.count("!") + content.count("?")
    structure_score = min(1.0, punctuation_count / 8.0)
    return (unique_token_score * 0.5) + (length_score * 0.35) + (structure_score * 0.15)


def _is_document_new_enough(doc: RawDocument, oldest_timestamp: int | None) -> bool:
    """Return True when a document is newer than the refresh watermark.

    If no watermark is provided or publish metadata is unavailable/invalid,
    the document is kept (fail-open).
    """
    if not isinstance(oldest_timestamp, int | float) or oldest_timestamp <= 0:
        return True

    metadata = doc.metadata or {}
    published_raw = (
        metadata.get("published_at")
        or metadata.get("published")
        or metadata.get("published_date")
        or metadata.get("date")
        or metadata.get("year")
        or metadata.get("publication_year")
        or metadata.get("timestamp")
    )
    if published_raw in (None, ""):
        return True

    try:
        if isinstance(published_raw, int | float):
            published_dt = datetime.fromtimestamp(float(published_raw), tz=UTC)
        elif isinstance(published_raw, str) and published_raw.isdigit() and len(published_raw) == 4:
            published_dt = datetime(int(published_raw), 1, 1, tzinfo=UTC)
        else:
            normalized = str(published_raw).replace("Z", "+00:00")
            published_dt = datetime.fromisoformat(normalized)
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=UTC)
        return published_dt.timestamp() >= float(oldest_timestamp)
    except Exception:
        return True


def _relevance_item_text(item: dict) -> str:
    """Build a stable short text for relevance embedding comparison."""
    title = str(item.get("title") or "").strip()
    content = str(item.get("content") or item.get("snippet") or "").strip()
    return f"{title} {content}".strip()


def _keyword_tokens(text: str) -> set[str]:
    """Tokenize keywords and add a compact alphanumeric form for brand-like terms."""
    tokens = _tokenize(text)
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(compact) >= 3:
        tokens.add(compact)
    return tokens


def _project_anchor_terms(
    intent: dict,
    *,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> set[str]:
    """Build strict topical anchors from intent + project text for social relevance."""
    raw_terms: list[str] = []
    for value in (
        *(intent.get("must_match_terms") or []),
        *(intent.get("entities") or []),
        *(intent.get("keywords") or []),
        *(intent.get("domain_terms") or []),
        social_filter,
        project_title,
        project_description,
    ):
        raw_terms.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "").replace("#", " ")))

    anchors: set[str] = set()
    for raw in raw_terms:
        token = str(raw or "").strip().lower().replace("_", " ")
        token = re.sub(r"\s+", " ", token).strip()
        if len(token) < 3:
            continue
        if token in _GENERIC_RELEVANCE_TERMS or token in _SOCIAL_BROAD_TERMS:
            continue
        if token in _PROJECT_TEXT_STOPWORDS:
            continue
        anchors.add(token)
    return anchors


def _build_relevance_terms(intent: dict, social_filter: str) -> set[str]:
    """Build keyword terms used to rank social accounts and items."""
    terms: set[str] = set()
    keywords: list[str] = intent.get("keywords") or []
    for kw in keywords:
        terms.update(_keyword_tokens(str(kw)))

    search_filters = intent.get("search_filters") or {}
    terms.update(_keyword_tokens(str(search_filters.get("instagram") or "")))
    terms.update(_keyword_tokens(str(social_filter or "")))

    # Remove low-signal generic tokens
    filtered = {t for t in terms if t and t not in _GENERIC_RELEVANCE_TERMS}
    return filtered


def _build_brand_terms(intent: dict) -> set[str]:
    """Extract non-generic brand/product tokens from intent keywords."""
    keywords: list[str] = intent.get("keywords") or []
    terms: set[str] = set()
    for kw in keywords:
        for tok in _keyword_tokens(str(kw)):
            if tok in _GENERIC_RELEVANCE_TERMS:
                continue
            terms.add(tok)
    return terms


def _project_context_query(project_title: str, project_description: str) -> str:
    """Create a short fallback query from project title/description."""
    combined = f"{project_title} {project_description}".strip()
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", combined.lower())
    seen = set()
    out = []
    for t in tokens:
        if t not in seen and t not in _PROJECT_TEXT_STOPWORDS:
            seen.add(t)
            out.append(t)
            if len(out) >= 5:
                break
    return " ".join(out) or "latest trends"


def _social_query_terms(
    intent: dict,
    *,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> list[str]:
    """Build deterministic high-signal query terms for social retrieval."""
    terms: list[str] = []
    seen: set[str] = set()

    def _push(tokens):
        for t in tokens:
            cleaned = str(t).strip().lower()
            if len(cleaned) >= 3 and cleaned not in seen and cleaned not in _PROJECT_TEXT_STOPWORDS:
                seen.add(cleaned)
                terms.append(cleaned)

    _push(re.findall(r"[a-zA-Z0-9_]+", social_filter.replace("#", " ")))
    _push(intent.get("must_match_terms") or [])
    _push(intent.get("entities") or [])
    _push(intent.get("keywords") or [])
    _push(re.findall(r"[a-zA-Z0-9_]+", _project_context_query(project_title, project_description)))

    return terms


def _prioritize_social_terms(intent: dict, terms: list[str], *, budget: int) -> list[str]:
    """Prioritize must-match/entities first, then fill with remaining terms."""
    must = {str(t).lower() for t in (intent.get("must_match_terms") or [])}
    entities = {str(t).lower() for t in (intent.get("entities") or [])}

    out = []
    # Pass 1: Must match
    for t in terms:
        if t in must:
            out.append(t)
    # Pass 2: Entities
    for t in terms:
        if t in entities and t not in must:
            out.append(t)
    # Pass 3: Fillers
    for t in terms:
        if t not in must and t not in entities:
            out.append(t)

    return out[:budget]


def _query_for_social(
    intent: dict,
    *,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Select a robust social search seed even when intent filters are sparse."""
    search_filters = intent.get("search_filters") or {}
    query_budget = max(4, int(settings.INGEST_SOCIAL_QUERY_MAX_TERMS))

    explicit_handles = [
        f"@{h.lstrip('@')}"
        for h in re.findall(
            r"@[A-Za-z0-9_]{2,20}",
            " ".join(
                [
                    str(social_filter or ""),
                    str(search_filters.get("social") or ""),
                    str(project_title or ""),
                    str(project_description or ""),
                ]
            ),
        )
    ]
    explicit_handles = list(dict.fromkeys(explicit_handles))[:2]

    direct = str(
        social_filter or search_filters.get("social") or search_filters.get("instagram") or ""
    ).strip()

    if direct:
        tokens = []
        for raw in re.findall(r"[a-zA-Z0-9_]+", direct.replace("#", " ")):
            token = str(raw or "").strip().lower().replace("_", " ")
            token = re.sub(r"\s+", " ", token).strip()
            if (
                len(token) < 3
                or token.isdigit()
                or token in _GENERIC_RELEVANCE_TERMS
                or token in _PROJECT_TEXT_STOPWORDS
            ):
                continue
            tokens.append(token)

        merged = list(dict.fromkeys(tokens))
        if len(merged) < 3:
            extras = _social_query_terms(
                intent,
                social_filter=social_filter,
                project_title=project_title,
                project_description=project_description,
            )
            for token in extras:
                if token not in merged:
                    merged.append(token)

        merged = _prioritize_social_terms(intent, merged, budget=query_budget)
        if merged:
            return " ".join(explicit_handles + merged).strip()

    terms = _social_query_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    terms = _prioritize_social_terms(intent, terms, budget=query_budget)
    if terms:
        return " ".join(explicit_handles + terms).strip()

    return " ".join(
        explicit_handles + [_project_context_query(project_title, project_description)]
    ).strip()


def _query_for_patents(
    intent: dict,
    *,
    source: str = "patentsview",
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Select the best available query string for patent tools."""
    search_filters = intent.get("search_filters") or {}
    entities = intent.get("entities") or []
    keywords = intent.get("keywords") or []

    direct = str(search_filters.get("patents") or search_filters.get("papers") or "").strip()
    if direct:
        return direct

    terms: list[str] = []
    if entities:
        terms.append(str(entities[0]))
    terms.extend(str(keyword) for keyword in keywords[:4] if str(keyword).strip())
    if terms:
        return " ".join(term.strip() for term in terms if term.strip())

    return _project_context_query(project_title, project_description)


def _looks_like_patent_url(url: str) -> bool:
    """Return True when a URL likely points to a patent record/document."""
    lowered = (url or "").strip().lower()
    if not lowered:
        return False
    parsed = urlparse(lowered)
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path or ""

    if host == "patents.google.com" and "/patent/" in path:
        return True
    if host == "patentscope.wipo.int" and ("/search/" in path or "/detail/" in path):
        return True
    if host == "worldwide.espacenet.com" and "/patent/" in path:
        return True
    if host == "lens.org" and "/patent/" in path:
        return True
    if host.endswith("uspto.gov") and "/patents/" in path:
        return True

    return host in {"patents.justia.com", "freepatentsonline.com"} and "/patent" in path


def _query_for_papers(
    intent: dict,
    *,
    source: str = "openalex",
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Select the best available paper query string with safe fallbacks."""
    search_filters = intent.get("search_filters") or {}
    entities = intent.get("entities") or []
    keywords = intent.get("keywords") or []

    direct = str(search_filters.get("papers") or "").strip()
    if direct:
        return direct

    news_fallback = str(search_filters.get("news") or "").strip()
    if news_fallback:
        return news_fallback

    terms = [str(value).strip() for value in entities[:2] if str(value).strip()]
    terms.extend(str(value).strip() for value in keywords[:6] if str(value).strip())
    if terms:
        return " ".join(terms)

    social_fallback = str(social_filter or "").replace("#", " ").strip()
    if social_fallback:
        return social_fallback

    return _project_context_query(project_title, project_description)


def _required_social_match_count(anchor_terms: set[str]) -> int:
    """Use OR semantics: one topical match is enough to keep a candidate."""
    _ = anchor_terms
    return 1


def _required_must_match_count(must_terms: list[str] | set[str] | None = None) -> int:
    """Use OR semantics for explicit must-match terms across all sources."""
    _ = must_terms
    return 1


def _query_for_news_or_web(intent: dict, *, source: str = "perigon") -> str:
    """Select the best available query string for news/web tools."""
    search_filters = intent.get("search_filters") or {}
    entities = intent.get("entities") or []
    keywords = intent.get("keywords") or []
    seed_query = (
        search_filters.get("news")
        or search_filters.get("papers")
        or (entities[0] if entities else "")
        or " ".join(str(k) for k in keywords[:4])
    )
    return str(seed_query or "").strip()


def _clean_term_values(values: list[str] | None) -> list[str]:
    """Clean and normalize a list of term values."""
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        term = str(raw or "").strip().lower().replace("_", " ")
        term = re.sub(r"\s+", " ", term).strip()
        if not term:
            continue
        out.append(term)
    return out


def _social_must_terms(intent: dict, query_terms: list[str]) -> list[str]:
    """Build strict social must-match terms from intent and query terms."""
    explicit = _clean_term_values(intent.get("must_match_terms") or [])
    if explicit:
        return explicit[:5]

    specificity = str(intent.get("query_specificity") or "").strip().lower()
    if specificity != "specific":
        return []

    candidates: list[str] = []
    for value in intent.get("entities") or []:
        candidates.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    for value in intent.get("keywords") or []:
        candidates.extend(re.findall(r"[a-zA-Z0-9_]+", str(value or "")))
    candidates.extend(query_terms)

    out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        term = str(raw or "").strip().lower().replace("_", " ")
        term = re.sub(r"\s+", " ", term).strip()
        if len(term) < 3:
            continue
        if term in seen:
            continue
        if term in _GENERIC_RELEVANCE_TERMS or term in _SOCIAL_BROAD_TERMS:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= 5:
            break
    return out


def _query_variants_for_source(intent: dict, base_query: str) -> list[str]:
    """Build bounded query variants: base + focused term/phrase queries from intent."""
    base = str(base_query or "").strip()
    if not base:
        return []
    max_variants = max(1, int(settings.INGEST_QUERY_VARIANTS_PER_SOURCE))
    out: list[str] = []
    seen: set[str] = set()

    def _push(query: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    _push(base)
    focused_phrases: list[str] = []
    for value in (
        *(intent.get("must_match_terms") or []),
        *(intent.get("entities") or []),
        *(intent.get("keywords") or []),
        *(intent.get("domain_terms") or []),
    ):
        phrase = re.sub(r"\s+", " ", str(value or "").replace("#", " ")).strip().lower()
        if not phrase:
            continue
        words = [w for w in re.findall(r"[a-zA-Z0-9_]+", phrase) if len(w) >= 3]
        if not words:
            continue
        normalized_words: list[str] = []
        for word in words:
            lowered = word.lower()
            if (
                lowered.isdigit()
                or lowered in _GENERIC_RELEVANCE_TERMS
                or lowered in _PROJECT_TEXT_STOPWORDS
            ):
                continue
            normalized_words.append(lowered)
        if not normalized_words:
            continue
        focused = " ".join(normalized_words[:4]).strip()
        if not focused:
            continue
        focused_phrases.append(focused)

    for focused in focused_phrases:
        _push(focused)
        if len(out) >= max_variants:
            break
        _push(f"{focused} {base}")
        if len(out) >= max_variants:
            break

    if len(out) < max_variants:
        token_pool: list[str] = []
        for phrase in focused_phrases:
            token_pool.extend(re.findall(r"[a-zA-Z0-9_]+", phrase))
        for raw in token_pool:
            token = str(raw or "").strip().lower()
            if (
                len(token) < 3
                or token.isdigit()
                or token in _GENERIC_RELEVANCE_TERMS
                or token in _PROJECT_TEXT_STOPWORDS
            ):
                continue
            _push(f"{base} {token}")
            if len(out) >= max_variants:
                break
    return out


def _social_query_variants(
    *,
    intent: dict,
    base_query: str,
    social_filter: str,
    project_title: str,
    project_description: str,
    expanded: bool = False,
) -> list[str]:
    """Build bounded social query variants with deterministic term-level fanout."""
    base = str(base_query or "").strip()
    if not base:
        return []

    max_variants = (
        max(1, int(settings.INGEST_SOCIAL_EXPANSION_MAX_VARIANTS))
        if expanded
        else max(1, int(settings.INGEST_SOCIAL_TIER1_MAX_VARIANTS))
    )
    probe_terms = max(0, int(settings.INGEST_SOCIAL_TIER1_PROBE_TERMS))
    out: list[str] = []
    seen: set[str] = set()

    def _push(query: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    _push(base)
    if len(out) >= max_variants:
        return out[:max_variants]

    budget = (
        max(8, int(settings.INGEST_SOCIAL_EXPANDED_QUERY_MAX_TERMS))
        if expanded
        else max(4, int(settings.INGEST_SOCIAL_QUERY_MAX_TERMS))
    )
    terms = _social_query_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    terms = _prioritize_social_terms(intent, terms, budget=budget)

    if not terms:
        for raw in re.findall(r"[a-zA-Z0-9_]+", base):
            token = str(raw or "").strip().lower()
            if (
                len(token) < 3
                or token in _GENERIC_RELEVANCE_TERMS
                or token in _PROJECT_TEXT_STOPWORDS
            ):
                continue
            terms.append(token)
            if len(terms) >= budget:
                break

    for term in terms[:probe_terms]:
        _push(term)
        if len(out) >= max_variants:
            return out[:max_variants]
        _push(f"{base} {term}")
        if len(out) >= max_variants:
            return out[:max_variants]

    if len(terms) >= 2:
        _push(f"{terms[0]} {terms[1]}")
        if len(out) >= max_variants:
            return out[:max_variants]
        _push(f"{base} {terms[0]} {terms[1]}")
        if len(out) >= max_variants:
            return out[:max_variants]

    if len(out) < max_variants:
        for variant in _query_variants_for_source(intent, base):
            _push(variant)
            if len(out) >= max_variants:
                break

    return out[:max_variants]


async def _run_source_with_timeout(
    source: str,
    coro,
    *,
    timeout_seconds: float | None = None,
) -> list[RawDocument]:
    """Run one source ingest with timeout and fail-open behavior."""
    import asyncio

    timeout = float(timeout_seconds or settings.INGEST_SOURCE_TIMEOUT)
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        logger.warning(
            "ingest_tool.timeout",
            source=source,
            timeout_seconds=timeout,
        )
        return []
    except Exception as exc:
        logger.warning("ingest_tool.failed", source=source, error=str(exc))
        return []


def _social_web_url_allowed(source: str, url: str) -> bool:
    """Return True when fallback URL is a concrete post/video URL for the target social source."""
    lowered_url = str(url or "").strip().lower()
    if not lowered_url:
        return False

    parsed = urlparse(lowered_url)
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path or ""

    if source == "social_youtube":
        return (
            host in {"youtube.com", "m.youtube.com"}
            and any(token in path for token in ("/watch", "/shorts/", "/live/"))
        ) or (host == "youtu.be" and len(path.strip("/")) >= 6)
    if source == "social_reddit":
        return host in {"reddit.com", "old.reddit.com"} and "/r/" in path and "/comments/" in path
    if source == "social_x":
        return host in {"x.com", "twitter.com", "mobile.twitter.com"} and "/status/" in path
    if source == "social_instagram":
        return host == "instagram.com" and any(
            path.startswith(prefix) for prefix in ("/reel/", "/p/")
        )
    if source == "social_tiktok":
        return host in {"tiktok.com", "m.tiktok.com"} and "/video/" in path
    return False


def _social_web_query(source: str, base_query: str) -> str:
    """Build a site-constrained web query for social fallback retrieval."""
    q = str(base_query or "").strip()
    if source == "social_youtube":
        return f"{q} site:youtube.com"
    if source == "social_reddit":
        return f"{q} site:reddit.com/r"
    if source == "social_x":
        return f"{q} (site:x.com OR site:twitter.com)"
    if source == "social_instagram":
        return f"{q} site:instagram.com/reel"
    if source == "social_tiktok":
        return f"{q} site:tiktok.com"
    return q


async def _social_web_fallback_docs(
    *,
    source: str,
    project_id: str,
    user_id: str,
    query: str,
    keep_limit: int,
) -> list[RawDocument]:
    """Fallback social ingestion from web search when native social APIs return no rows."""
    constrained_query = _social_web_query(source, query)
    if not constrained_query:
        return []

    candidates: list[dict] = []
    try:
        candidates.extend(await search_exa(project_id, constrained_query))
    except Exception:
        logger.exception("social_web_fallback.exa_failed", source=source, project_id=project_id)
    try:
        candidates.extend(await search_tavily(project_id, constrained_query))
    except Exception:
        logger.exception("social_web_fallback.tavily_failed", source=source, project_id=project_id)

    docs: list[RawDocument] = []
    seen_urls: set[str] = set()
    for row in candidates:
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        if not _social_web_url_allowed(source, url):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = str(row.get("title") or "").strip() or "Social result"
        content = str(row.get("content") or row.get("snippet") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source=source,
                source_id=url,
                title=title[:240],
                content=content,  # Store full content (chunker handles chunking)
                metadata={
                    "platform": source.removeprefix("social_"),
                    "url": url,
                    "published_at": row.get("published_date") or "",
                    "tool": f"{row.get('tool') or 'web'}_social_fallback",
                    "fallback_origin": True,
                },
            )
        )
        if len(docs) >= max(1, keep_limit):
            break

    logger.info(
        "social_web_fallback.done",
        source=source,
        project_id=project_id,
        query_preview=constrained_query[:100],
        candidate_count=len(candidates),
        doc_count=len(docs),
    )
    return docs


async def _noop() -> list[RawDocument]:
    """Return an empty source result for disabled gather slots."""
    return []


def _build_source_counts(documents: list[RawDocument]) -> dict[str, int]:
    """Build per-source counters from kept documents (single pass)."""
    from collections import Counter

    return dict(Counter(doc.source for doc in documents))


def _build_expanded_social_filter(
    intent: dict,
    *,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
) -> str:
    """Build a broader social search query for expansion when initial results are sparse."""
    terms = _social_query_terms(
        intent,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    # Filter out very common/stop words and keep high-signal ones
    signal_terms = [t for t in terms if len(t) > 3]
    if not signal_terms:
        return social_filter or project_title
    return " ".join(signal_terms[: settings.INGEST_SOCIAL_EXPANDED_QUERY_MAX_TERMS])


def _extract_expansion_links(docs: list[RawDocument]) -> dict[str, set[str]]:
    """Extract authors and URLs from social documents to seed expansion searches."""
    expansion: dict[str, set[str]] = {
        "social_instagram": set(),
        "social_tiktok": set(),
        "social_youtube": set(),
        "social_reddit": set(),
        "social_x": set(),
    }
    for doc in docs:
        metadata = doc.metadata or {}
        author = str(metadata.get("author") or "").strip()
        if author:
            # Seed author search in the same platform
            expansion[doc.source].add(author)

        content = doc.content or ""
        # Find URLs that might point to other social platforms
        urls = re.findall(r"https?://[^\s<>\"']+", content)
        for url in urls:
            lowered = url.lower()
            if "instagram.com" in lowered:
                expansion["social_instagram"].add(url)
            elif "tiktok.com" in lowered:
                expansion["social_tiktok"].add(url)
            elif "youtube.com" in lowered or "youtu.be" in lowered:
                expansion["social_youtube"].add(url)
            elif "reddit.com" in lowered:
                expansion["social_reddit"].add(url)
            elif "twitter.com" in lowered or "x.com" in lowered:
                expansion["social_x"].add(url)
    return expansion
