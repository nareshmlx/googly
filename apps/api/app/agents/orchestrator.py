"""OrchestratorAgent — coordinates the full query pipeline.

This is not an Agno Team (no routing logic needed at this layer).
It is a thin async coordinator that:
  1. Runs IntentAgent to get structured query intent
  2. Uses ProjectRelevanceAgent to decide which projects to search
  3. Queries the KB retriever across the relevant projects
  4. If KB scores are all below threshold, falls back to a general LLM response
  5. Streams the SynthesisAgent response token-by-token

Keeping this as a plain async function (not an Agno Team) because:
- All orchestration logic is deterministic and sequential/parallel where it should be
- Agno Team routing adds overhead and non-determinism we don't need here
- The agents themselves (Intent, Relevance, Synthesis) are thin wrappers around LLM calls

Called from ChatService. Yields SSE-formatted strings.
"""

import asyncio
import math
import re
import time
from collections import Counter
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from functools import cache
from typing import Any

import orjson
import structlog
from agno.run.agent import RunContentEvent

from app.agents.intent import build_intent_agent
from app.agents.patent import build_patent_agent
from app.agents.project_relevance import build_project_relevance_agent
from app.agents.query_enhancer import build_query_enhancer_agent
from app.agents.synthesis import build_synthesis_agent
from app.core.cache_keys import stable_hash
from app.core.config import settings
from app.core.dedup_papers import deduplicate_papers
from app.core.metrics import (
    exa_usage_rate,
    orchestrator_stage_duration_seconds,
    research_deduplication_rate,
    research_tool_calls_total,
)
from app.core.normalize import coerce_int
from app.core.paper_schema import normalize_paper_schema
from app.core.query_policy import lexical_entity_coverage
from app.core.redis import get_redis
from app.core.utils import dedup_terms
from app.kb.retriever import retrieve
from app.tools.news_perigon import search_perigon
from app.tools.papers_arxiv import search_arxiv
from app.tools.papers_pubmed import search_pubmed
from app.tools.papers_semantic_scholar import search_semantic_scholar
from app.tools.search_exa import search_exa
from app.tools.search_tavily import search_tavily

logger = structlog.get_logger(__name__)


# Module-level singletons — initialised once on first call, reused across all requests
# (Agno agents are stateless per arun() call — safe to share across requests)


@cache
def _get_intent_agent():
    return build_intent_agent()


@cache
def _get_query_enhancer_agent():
    return build_query_enhancer_agent()


@cache
def _get_synthesis_agent():
    return build_synthesis_agent()


@cache
def _get_project_relevance_agent():
    return build_project_relevance_agent()


@cache
def _get_patent_agent():
    return build_patent_agent()


def _safe_int(value: object) -> int:
    """Convert mixed numeric values (int/float/str) to int safely."""
    return coerce_int(value)


def _extract_social_insights(kb_results: list[dict]) -> dict:
    """
    Build compact social evidence summary from retrieved KB chunks.

    This gives synthesis a grounded structure instead of raw caption-only context.
    """
    social_chunks = [
        c
        for c in kb_results
        if c.get("source")
        in {"social_instagram", "social_tiktok", "social_youtube", "social_reddit", "social_x"}
    ]
    if not social_chunks:
        return {}

    platform_counts: Counter[str] = Counter()
    creator_counts: Counter[str] = Counter()
    top_posts: list[dict] = []
    newest_ts: int | None = None
    oldest_ts: int | None = None

    for chunk in social_chunks:
        metadata = chunk.get("metadata") or {}
        source = chunk.get("source")
        platform = (
            "instagram"
            if source == "social_instagram"
            else "tiktok"
            if source == "social_tiktok"
            else "youtube"
            if source == "social_youtube"
            else "reddit"
            if source == "social_reddit"
            else "x"
        )
        platform_counts[platform] += 1

        author = (
            metadata.get("username")
            or metadata.get("author_username")
            or (chunk.get("title") or "").lstrip("@")
            or "unknown"
        )
        author = str(author)
        creator_counts[author] += 1

        likes = _safe_int(metadata.get("like_count") or metadata.get("likes"))
        views = _safe_int(metadata.get("view_count") or metadata.get("views"))
        engagement = likes + (views // 10)
        caption = str(chunk.get("content") or "").strip()
        top_posts.append(
            {
                "platform": platform,
                "author": author,
                "likes": likes,
                "views": views,
                "engagement": engagement,
                "caption_preview": caption[:120],
            }
        )

        ts_val = metadata.get("timestamp")
        ts_int = _safe_int(ts_val)
        if ts_int > 0:
            newest_ts = ts_int if newest_ts is None else max(newest_ts, ts_int)
            oldest_ts = ts_int if oldest_ts is None else min(oldest_ts, ts_int)

    top_creators = [{"author": a, "posts": n} for a, n in creator_counts.most_common(5)]

    top_posts.sort(key=lambda x: x["engagement"], reverse=True)
    top_posts = top_posts[:5]

    freshness_days: int | None = None
    if newest_ts:
        now_ts = int(datetime.now(UTC).timestamp())
        freshness_days = max(0, math.floor((now_ts - newest_ts) / 86400))

    return {
        "social_chunk_count": len(social_chunks),
        "platform_counts": platform_counts,
        "top_creators": top_creators,
        "top_posts": top_posts,
        "freshness_days": freshness_days,
    }


def _looks_like_research_query(query: str) -> bool:
    """Heuristic fallback for research intent when intent classification is weak."""
    q = query.lower()
    return bool(
        re.search(
            r"\b(research|paper|papers|study|studies|journal|literature|citation|citations|academic|scholarly)\b",
            q,
        )
    )


def _format_openalex_context(papers: list[dict]) -> str:
    """Format OpenAlex papers into synthesis context blocks with clickable citation links."""
    parts: list[str] = []
    for paper in papers:
        title = str(paper.get("title") or "Untitled paper")
        paper_id = str(paper.get("paper_id") or "")
        year = paper.get("publication_year") or "n/a"
        # Try both field names — legacy OpenAlex uses `abstract`, tools use `content`
        abstract = str(paper.get("abstract") or paper.get("content") or "").strip()
        summary = abstract[:1200] if abstract else "Abstract unavailable."
        doi = str(paper.get("doi") or "")

        # Build citation link — prefer DOI, fallback to OpenAlex ID
        if doi:
            # Strip "https://doi.org/" prefix if present to get clean DOI
            clean_doi = doi.replace("https://doi.org/", "")
            citation_url = f"https://doi.org/{clean_doi}"
            source_header = f"[{title}]({citation_url}) (paper, {year})"
        elif paper_id:
            # OpenAlex ID format: https://openalex.org/W123456789
            citation_url = (
                f"https://openalex.org/{paper_id}" if not paper_id.startswith("http") else paper_id
            )
            source_header = f"[{title}]({citation_url}) (paper, {year})"
        else:
            source_header = f"[Source: {title} (paper, {year})]"

        parts.append(f"{source_header}\n{summary}")
    return "\n\n---\n\n".join(parts)


def _build_academic_query(query: str, intent: dict) -> str:
    """Build a focused OpenAlex query while stripping meta/filler phrasing."""
    intent = intent or {}
    meta_terms = frozenset(
        {
            "research papers",
            "research paper",
            "papers",
            "paper",
            "studies",
            "study",
            "articles",
            "article",
            "publications",
            "publication",
            "latest",
            "recent",
            "newest",
            "current",
        }
    )

    entity_source = (intent.get("must_match_terms") or []) + (intent.get("entities") or [])
    entities: list[str] = [
        e.strip()
        for e in entity_source
        if isinstance(e, str) and e.strip() and e.strip().lower() not in meta_terms
    ]
    if entities:
        text = " ".join(entities[:5])
        lowered = text.lower()
        domain = str(intent.get("domain") or "").lower()
        if domain in {
            "cosmetics",
            "beauty_market_intelligence",
            "fragrance",
            "skincare",
        } and not any(t in lowered for t in {"beauty", "cosmetics", "skincare", "fragrance"}):
            text = f"{text} skincare cosmetics"
        return re.sub(r"\s+", " ", text).strip()

    text = str(query or "")
    for marker in ("\n\nAnswer", "\nAnswer", "\n\nEvidence", "\nEvidence"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break
    text = re.sub(
        r"^(what|which|who|where|when|how|why|tell me|show me|find|list|give me)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"\b(are|is|do|does|latest|recent|newest|current|research|papers|paper|studies|study|about|on|for)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 220:
        text = text[:220].rstrip()

    lowered = text.lower()
    if not any(
        term in lowered for term in {"beauty", "cosmetics", "skincare", "makeup", "fragrance"}
    ):
        domain = str(intent.get("domain") or "").lower()
        if domain in {"cosmetics", "beauty_market_intelligence", "fragrance", "skincare"}:
            text = f"{text} skincare cosmetics".strip()
    return text


def build_academic_query(query: str, intent: dict) -> str:
    """Canonical academic query builder used by orchestrator retrieval."""
    return _build_academic_query(query, intent)


def _clean_user_query(query: str) -> str:
    """Trim accidental pasted answer/evidence blocks before orchestration."""
    text = str(query or "")
    for marker in ("\n\nAnswer", "\nAnswer", "\n\nEvidence", "\nEvidence"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break
    text = re.sub(r"\s+", " ", text).strip()
    return text or str(query or "").strip()


def _format_search_fallback_context(results: list[dict]) -> str:
    """
    Format search results from multiple sources into synthesis context.

    Handles results from Tavily, Exa, and Perigon with source attribution.
    """
    if not results:
        return ""

    parts: list[str] = []
    for result in results[:15]:  # Limit to top 15 across all sources
        title = str(result.get("title") or "Untitled").strip()
        url = str(result.get("url") or "").strip()
        content = str(result.get("content") or "").strip()
        source = str(result.get("source") or "unknown").strip()

        if not content:
            continue

        # Truncate content to max 1500 chars per result — preserves Exa's richer snippets
        if len(content) > 1500:
            content = f"{content[:1500]}..."

        # Embed URL directly in the source header as a markdown link so the synthesis
        # agent can copy it verbatim for inline citations without a separate URL: line.
        if url and title != "Untitled":
            source_header = f"[{title}]({url})"
        elif url:
            source_header = f"[{source}]({url})"
        else:
            source_header = f"[{title} ({source})]" if title != "Untitled" else f"[{source}]"
        parts.append(f"{source_header}\n{content}")

    return "\n\n---\n\n".join(parts)


def _format_kb_context(kb_results: list[dict]) -> str:
    """
    Format KB chunks into synthesis context blocks with clickable citation links.

    Handles social media metadata and academic identifiers (DOI, PMID, arXiv).
    Surfaces URLs to ensure the synthesis agent can cite specific sources.
    """
    if not kb_results:
        return ""

    context_parts = []
    urls_found = 0
    urls_missing = 0

    for chunk in kb_results:
        title = chunk.get("title") or "Untitled"
        source = chunk.get("source", "unknown")
        score = float(chunk.get("score", 0.0) or 0.0)
        relevance = f"{score:.2f}"
        metadata = chunk.get("metadata") or {}
        content = str(chunk.get("content") or "")

        if source in {
            "social_instagram",
            "social_tiktok",
            "social_youtube",
            "social_reddit",
            "social_x",
        }:
            likes = _safe_int(metadata.get("like_count") or metadata.get("likes"))
            views = _safe_int(metadata.get("view_count") or metadata.get("views"))
            author = (
                metadata.get("username")
                or metadata.get("author_username")
                or metadata.get("author")
                or str(title).lstrip("@")
            )
            # Extract URL for social media sources — enables citations
            social_url = (
                metadata.get("url") or metadata.get("source_url") or metadata.get("link") or ""
            ).strip()

            if social_url:
                urls_found += 1
                # Format with clickable link for proper citation
                source_header = f"[{title}]({social_url}) ({source}, relevance: {relevance})"
                context_parts.append(
                    f"{source_header}\nauthor={author} likes={likes} views={views}\n{content}"
                )
            else:
                urls_missing += 1
                # Fallback if no URL available
                context_parts.append(
                    f"[Source: {title} ({source}, relevance: {relevance})]\n"
                    f"author={author} likes={likes} views={views}\n"
                    f"{content}"
                )
        else:
            # Surface any URL stored at ingest time so the synthesis agent can
            # cite specific pages rather than falling back to homepage guesses.
            doi = str(metadata.get("doi") or "").strip()
            pmid = str(metadata.get("pmid") or "").strip()
            arxiv_id = str(metadata.get("arxiv_id") or "").strip()
            chunk_url = (
                metadata.get("url") or metadata.get("source_url") or metadata.get("link") or ""
            ).strip()

            # Fallback: use open_access_url / pdf_url stored by OpenAlex ingest when
            # the primary url field is empty (common for paywalled papers with no DOI URL).
            if not chunk_url:
                chunk_url = str(
                    metadata.get("open_access_url") or metadata.get("pdf_url") or ""
                ).strip()

            # Fallback: use paper_id to construct a citation link.
            # OpenAlex stores a full URL (https://openalex.org/W...) — use directly.
            # Semantic Scholar stores a bare hash (paperId) — construct the SS paper URL.
            if not doi and not chunk_url:
                paper_id = str(metadata.get("paper_id") or "").strip()
                if paper_id:
                    if paper_id.startswith("http"):
                        chunk_url = paper_id
                    else:
                        # Bare Semantic Scholar paperId — build the canonical paper URL
                        chunk_url = f"https://www.semanticscholar.org/paper/{paper_id}"

            # Prioritize clickable links for synthesis agent
            if chunk_url:
                final_url = chunk_url
                urls_found += 1
            elif doi:
                if doi.startswith("http"):
                    final_url = doi  # Already a full URL
                else:
                    final_url = f"https://doi.org/{doi.replace('https://doi.org/', '')}"
                urls_found += 1
            elif pmid:
                final_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                urls_found += 1
            elif arxiv_id:
                final_url = f"https://arxiv.org/abs/{arxiv_id}"
                urls_found += 1
            else:
                final_url = ""
                urls_missing += 1

            if final_url and title != "Untitled":
                source_header = f"[{title}]({final_url}) ({source}, relevance: {relevance})"
            elif final_url:
                source_header = f"[{source}]({final_url}) (relevance: {relevance})"
            elif title != "Untitled":
                # No URL available - use plain text citation but still label clearly
                source_header = f"[Source: {title} ({source}, relevance: {relevance})]"
            else:
                source_header = f"[Source: {source} (relevance: {relevance})]"
            context_parts.append(f"{source_header}\n{content}")

    # Log URL availability for debugging citation issues
    logger.info(
        "orchestrator.kb_context_formatted",
        total_chunks=len(kb_results),
        urls_found=urls_found,
        urls_missing=urls_missing,
    )

    return "\n\n---\n\n".join(context_parts)


def _asks_for_paper_list(query: str) -> bool:
    """Return True when user explicitly asks for papers/studies/literature results."""
    q = query.lower()
    return bool(
        re.search(
            r"\b(paper|papers|study|studies|journal|literature|citations?|academic|scholarly)\b",
            q,
        )
    )


_PAPER_SUMMARY_STOPWORDS: set[str] = {
    "about",
    "across",
    "analysis",
    "beauty",
    "between",
    "cosmetic",
    "cosmetics",
    "effect",
    "effects",
    "from",
    "into",
    "latest",
    "paper",
    "papers",
    "research",
    "review",
    "role",
    "study",
    "towards",
    "using",
    "with",
}

# Compiled once at module load — matches unambiguous "search the web" intent phrases.
# Word boundaries (\b) prevent false positives on research phrases like
# "recent studies" or "current treatment" that should still use KB.
_FORCE_WEB_PATTERNS = re.compile(
    r"\b("
    r"search\s+(?:the\s+)?web"
    r"|search\s+online"
    r"|look\s+up\s+online"
    r"|find\s+online"
    r"|check\s+online"
    r"|google\s+(?:for\s+)?\w+"
    r")\b",
    re.IGNORECASE,
)


def _paper_context_snippet(paper: dict, max_chars: int = 420) -> str:
    """Return concise context from abstract/content/title (1-2 sentences, bounded length).

    All research tools emit the body text under the key ``content``, while the
    legacy OpenAlex path uses ``abstract``.  Try both so neither path is blank.
    """
    abstract = str(paper.get("abstract") or paper.get("content") or "").strip()
    text = abstract if abstract else str(paper.get("title") or "").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return "Context unavailable."

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]
    snippet = " ".join(sentences[:2]).strip()
    if len(snippet) > max_chars:
        return f"{snippet[:max_chars].rstrip()}..."
    return snippet


def _overall_papers_summary(papers: list[dict]) -> str:
    """Build a short, deterministic summary across selected papers."""
    if not papers:
        return "No paper-level evidence was available to summarize."

    years: list[int] = []
    for paper in papers:
        publication_year = paper.get("publication_year")
        if isinstance(publication_year, int) and publication_year > 0:
            years.append(publication_year)
    citations = [_safe_int(p.get("cited_by_count")) for p in papers]
    avg_citations = round(sum(citations) / len(citations), 1) if citations else 0.0

    token_counts: dict[str, int] = {}
    for paper in papers:
        title = str(paper.get("title") or "")
        # Try both field names — legacy OpenAlex uses `abstract`, tools use `content`
        abstract = str(paper.get("abstract") or paper.get("content") or "")
        for token in re.findall(r"[a-z0-9]+", f"{title} {abstract}".lower()):
            if len(token) < 4 or token in _PAPER_SUMMARY_STOPWORDS:
                continue
            token_counts[token] = token_counts.get(token, 0) + 1

    top_terms = [
        token for token, _ in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    ]
    theme_text = ", ".join(top_terms) if top_terms else "beauty product science and consumer trends"

    year_text = f"{min(years)}-{max(years)}" if years else "mixed publication years"

    return (
        f"Across {len(papers)} papers ({year_text}), recurring themes include {theme_text}. "
        f"Average citation count in this set is {avg_citations}."
    )


def _format_openalex_paper_list_answer(query: str, papers: list[dict]) -> str:
    """Build a deterministic paper-list response from OpenAlex results."""
    if not papers:
        return (
            "No relevant papers were retrieved from OpenAlex for this query. "
            "Please try a more specific beauty topic (for example: sunscreen filters, "
            "retinol stability, fragrance allergens)."
        )

    max_items = min(10, len(papers))
    lines: list[str] = [
        f'Found {len(papers)} relevant papers from OpenAlex for: "{query}".',
        "",
        "Latest Papers",
    ]
    for idx, paper in enumerate(papers[:max_items], start=1):
        title = str(paper.get("title") or "Untitled paper")
        year = paper.get("publication_year") or "n/a"
        doi = str(paper.get("doi") or "").strip()
        paper_id = str(paper.get("paper_id") or "").strip()
        # `url` is a last-resort fallback — normalization should have already
        # promoted it into paper_id, but this guards against any missed edge case.
        url_fallback = str(paper.get("url") or "").strip()

        # Build a clean clickable link in markdown format
        raw_link = doi or paper_id or url_fallback
        if raw_link:
            # Ensure DOI is a full URL
            if raw_link and not raw_link.startswith("http"):
                raw_link = f"https://doi.org/{raw_link}"
            link_md = f"[{raw_link}]({raw_link})"
        else:
            link_md = "Link unavailable"

        citations = _safe_int(paper.get("cited_by_count"))
        context = _paper_context_snippet(paper)
        lines.append(f"{idx}. **{title}** ({year})")
        lines.append(f"   - {link_md}")
        if citations > 0:
            lines.append(f"   - Context: {context} (Citations: {citations})")
        else:
            lines.append(f"   - Context: {context}")

    if len(papers) > max_items:
        lines.append("")
        lines.append(f"Showing top {max_items} of {len(papers)} retrieved papers.")

    lines.append("")
    lines.append("Overall Summary")
    lines.append(_overall_papers_summary(papers[:max_items]))

    return "\n".join(lines)


async def _enhance_query_pre_intent(query: str) -> dict[str, Any]:
    """Normalize raw query before intent extraction while preserving specific terms."""
    default = {
        "rewritten_query": str(query or "").strip(),
        "must_match_terms": [],
        "expanded_terms": [],
        "domain_terms": [],
        "query_specificity": "broad",
        "target_domain": None,
    }
    default_typed: dict[str, Any] = default
    if not settings.QUERY_ENABLE_PRE_INTENT_ENHANCER:
        return default_typed
    if not settings.OPENAI_API_KEY:
        return default_typed

    try:
        enhancer = _get_query_enhancer_agent()
        result = await enhancer.arun(query)
        text = result.content if result and result.content else ""
        if not isinstance(text, str) or not text.strip():
            return default_typed
        parsed = orjson.loads(text.strip())
        rewritten_query = str(parsed.get("rewritten_query") or default_typed["rewritten_query"]).strip()
        must_terms = parsed.get("must_match_terms") or []
        if not isinstance(must_terms, list):
            must_terms = []
        must_terms = [str(term).strip() for term in must_terms if str(term).strip()][:5]
        domain_terms = parsed.get("domain_terms") or []
        if not isinstance(domain_terms, list):
            domain_terms = []
        domain_terms = [str(term).strip() for term in domain_terms if str(term).strip()][:4]
        specificity = str(parsed.get("query_specificity") or "broad").strip().lower()
        if specificity not in {"specific", "broad"}:
            specificity = "broad"
        return {
            "rewritten_query": rewritten_query or default["rewritten_query"],
            "must_match_terms": must_terms,
            "expanded_terms": parsed.get("expanded_terms") or [],
            "domain_terms": parsed.get("domain_terms") or [],
            "query_specificity": str(parsed.get("query_specificity") or "broad").lower(),
        }
    except orjson.JSONDecodeError:
        logger.warning("orchestrator.pre_intent_parse_error")
    except Exception:
        logger.exception("orchestrator.pre_intent_error")
    return default_typed


def _max_coverage_for_kb(kb_results: list[dict], must_match_terms: list[str]) -> float:
    """Compute best lexical entity coverage ratio across KB chunks."""
    if not kb_results:
        return 0.0
    best = 0.0
    for chunk in kb_results:
        text = f"{chunk.get('title', '')} {chunk.get('content', '')}"
        coverage = lexical_entity_coverage(text, must_match_terms)
        if coverage > best:
            best = coverage
    return best


def _filter_papers_by_entity_coverage(
    papers: list[dict], must_match_terms: list[str]
) -> list[dict]:
    """Keep papers with sufficient lexical coverage for specific queries."""
    if not papers or not must_match_terms:
        return papers
    threshold = settings.QUERY_ENTITY_MATCH_THRESHOLD
    scored: list[tuple[float, dict]] = []
    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('content', '')} {paper.get('abstract', '')}"
        coverage = lexical_entity_coverage(text, must_match_terms)
        scored.append((coverage, paper))
    kept = [paper for coverage, paper in scored if coverage >= threshold]
    if kept:
        return kept
    scored.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in scored[: max(1, min(5, len(scored)))]]


async def _extract_intent(query: str) -> dict[str, Any]:
    """
    Run IntentAgent and parse the JSON response.

    Results are cached in Redis for 1 hour keyed on the normalised query.
    Identical queries (e.g. repeated user searches) skip the LLM entirely,
    saving ~7.5 s per cache hit.

    Returns a default intent dict on any failure — never raises — so a bad
    intent classification degrades gracefully instead of crashing the pipeline.
    """
    default = {
        "domain": "general",
        "query_type": "general",
        "entities": [],
        "must_match_terms": [],
        "expanded_terms": [],
        "domain_terms": [],
        "query_specificity": "broad",
        "confidence": 0.5,
        "is_research_query": False,
    }
    default_typed: dict[str, Any] = default

    # Normalise query for cache key (lowercase + strip whitespace)
    query_normalised = query.lower().strip()
    cache_key = f"intent:{stable_hash([query_normalised])}"

    # --- Cache read ---
    redis = None
    try:
        redis = await get_redis()
        cached_raw = await redis.get(cache_key)
        if cached_raw:
            cached = orjson.loads(cached_raw)
            logger.info("orchestrator.intent.cache_hit", cache_key=cache_key)
            return cached
    except Exception:
        # Redis miss or failure — fall through to LLM call
        logger.warning("orchestrator.intent.cache_read_error", cache_key=cache_key, exc_info=True)

    # --- LLM call ---
    try:
        enhanced = await _enhance_query_pre_intent(query)
        intent_query = str(enhanced.get("rewritten_query") or query).strip() or query
        agent = _get_intent_agent()
        result = await agent.arun(intent_query)
        text = result.content if result and result.content else ""
        if isinstance(text, str) and text.strip():
            parsed = orjson.loads(text.strip())
            intent: dict[str, Any] = {**default_typed, **parsed}
            # Merge pre-intent enhancer constraints as a guardrail against
            # losing specific user entities during intent extraction.
            enhancer_terms = enhanced.get("must_match_terms") or []
            if enhancer_terms and isinstance(enhancer_terms, list):
                intent["must_match_terms"] = dedup_terms(
                    enhancer_terms, intent.get("must_match_terms"), limit=5
                )
            enhancer_expanded = enhanced.get("expanded_terms") or []
            if isinstance(enhancer_expanded, list):
                intent["expanded_terms"] = dedup_terms(
                    enhancer_expanded, intent.get("expanded_terms"), limit=8
                )
            enhancer_domain_terms = enhanced.get("domain_terms") or []
            if isinstance(enhancer_domain_terms, list):
                merged_domain = dedup_terms(enhancer_domain_terms, limit=4)
                if merged_domain:
                    intent["domain_terms"] = merged_domain
            if str(intent.get("query_specificity") or "").lower() not in {"specific", "broad"}:
                intent["query_specificity"] = str(
                    enhanced.get("query_specificity") or "broad"
                ).lower()

            # --- Cache write (only if Redis is available) ---
            if redis is not None:
                try:
                    await redis.setex(cache_key, settings.INTENT_CACHE_TTL, orjson.dumps(intent))
                    logger.info("orchestrator.intent.cache_write", cache_key=cache_key)
                except Exception:
                    # Cache write failure is non-fatal — result still returned to caller
                    logger.warning(
                        "orchestrator.intent.cache_write_error", cache_key=cache_key, exc_info=True
                    )

            return intent
    except orjson.JSONDecodeError:
        logger.warning("orchestrator.intent_parse_error", query_preview=query[:60])
    except Exception:
        logger.exception("orchestrator.intent_error")
    return default


async def _get_relevant_project_ids(
    query: str,
    primary_project_id: str,
    all_projects: list[dict],
) -> list[str]:
    """
    Use ProjectRelevanceAgent to pick which projects to search.

    Skips the LLM call when the user only has one project — saves a round-trip.
    Falls back to [primary_project_id] on any parsing or agent failure so the
    pipeline always searches at least one project.
    """
    if len(all_projects) <= 1:
        return [primary_project_id]

    projects_json = orjson.dumps(
        [
            {"id": p["id"], "title": p.get("title", ""), "description": p.get("description", "")}
            for p in all_projects
        ],
    ).decode("utf-8")
    prompt = f"query: {query}\n\nprojects:\n{projects_json}"

    try:
        agent = _get_project_relevance_agent()
        result = await agent.arun(prompt)
        text = result.content if result and result.content else ""
        if isinstance(text, str) and text.strip():
            parsed = orjson.loads(text.strip())
            relevant_ids = parsed.get("relevant_project_ids") or []
            if isinstance(relevant_ids, list) and relevant_ids:
                # Validate returned IDs exist in all_projects
                valid_ids = {p["id"] for p in all_projects}
                filtered = [pid for pid in relevant_ids if pid in valid_ids]
                if filtered:
                    logger.info(
                        "orchestrator.relevance.selected",
                        count=len(filtered),
                        query_preview=query[:60],
                    )
                    return filtered
    except orjson.JSONDecodeError:
        logger.warning("orchestrator.relevance_parse_error", query_preview=query[:60])
    except Exception:
        logger.exception("orchestrator.relevance_error")

    return [primary_project_id]


async def run_query(
    query: str,
    primary_project_id: str,
    all_projects: list[dict],
    user_id: str,
    session_id: str,
    openalex_enabled: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Full query pipeline: intent → project relevance → KB retrieve → synthesise → stream.

    Yields SSE-formatted strings: `data: {"token": "..."}\n\n`
    Terminates with: `data: [DONE]\n\n`
    """
    cleaned_query = _clean_user_query(query)
    if cleaned_query != (query or "").strip():
        logger.info(
            "orchestrator.query_sanitized",
            original_preview=(query or "")[:120],
            cleaned_preview=cleaned_query[:120],
        )

    logger.info(
        "orchestrator.start",
        user_id=user_id,
        primary_project_id=primary_project_id,
        project_count=len(all_projects),
        query_preview=cleaned_query[:80],
    )

    # Step 1 & 2: Intent extraction + Project relevance in PARALLEL
    # Task 2.1 improvement: 12.2s → 7.7s (4.5s savings)
    intent_result, relevance_result = await asyncio.gather(
        _extract_intent(cleaned_query),
        _get_relevant_project_ids(cleaned_query, primary_project_id, all_projects),
        return_exceptions=True,
    )

    # Graceful degradation: if one sub-task fails, use defaults for it
    if isinstance(intent_result, BaseException):
        logger.warning("orchestrator.intent_extraction_failed", error=str(intent_result))
        intent: dict[str, Any] = {"query_type": "general", "is_research_query": False}
    else:
        intent = {**intent_result}
    if isinstance(relevance_result, BaseException):
        logger.warning("orchestrator.relevance_extraction_failed", error=str(relevance_result))
        relevant_project_ids = [primary_project_id]
    else:
        relevant_project_ids = relevance_result

    query_type = intent.get("query_type", "general")
    is_research_query = bool(intent.get("is_research_query", False)) or _looks_like_research_query(
        cleaned_query
    )
    # Include papers for all queries if OpenAlex is enabled — let relevance score
    # determine if they're shown. This allows papers/patents to be cited even for
    # non-research queries (e.g. "acne treatments" might have relevant dermatology papers).
    # NOTE: KB_SCORE_THRESHOLD was intentionally lowered (0.70 → 0.40) alongside this
    # change. Together they make KB + papers more inclusive. Monitor for context noise
    # on non-research queries if answer quality degrades.
    exclude_papers = not openalex_enabled

    # Extract optional domain filter from intent (set when user names a specific
    # website or publication, e.g. "what does TechCrunch say about retinol?").
    # Only applied to general web search tools (Tavily, Exa web fallback) — never
    # to academic tools (arXiv, PubMed, Semantic Scholar) which don't support it.
    target_domain: str | None = intent.get("target_domain")
    include_domains: list[str] | None = [target_domain] if target_domain else None

    # Detect explicit web search request (user wants fresh/live web results, not KB).
    # _FORCE_WEB_PATTERNS is compiled at module level — see constant definition above.
    force_web_search = bool(_FORCE_WEB_PATTERNS.search(cleaned_query))

    logger.info(
        "orchestrator.intent_and_relevance",
        domain=intent.get("domain"),
        query_type=query_type,
        entities=intent.get("entities"),
        must_match_terms=intent.get("must_match_terms"),
        query_specificity=intent.get("query_specificity"),
        is_research_query=is_research_query,
        exclude_papers=exclude_papers,
        force_web_search=force_web_search,
        relevant_project_ids=relevant_project_ids,
        target_domain=target_domain,
    )

    # Step 1a: Route to specialized agents based on intent.
    # Skip specialized routing when target_domain is set — the user wants content
    # from a specific site (e.g. Fragrantica), so domain-filtered web search via
    # Tavily/Exa is more appropriate than Perigon or PatentsView which cannot
    # filter by domain.
    #
    # NOTE: TrendAgent removed - trend/social queries now use KB + Fallback + Synthesis
    # Project creation already ingests social data to KB, so runtime queries use that data.
    # If KB is empty, the fallback (Tavily, Exa, Perigon) will handle it.
    # This saves significant EnsembleData credits at runtime.

    # NOTE: PatentAgent is NOT called here — patent queries flow through KB first,
    # then web search fallback (Tavily/Exa/Perigon), exactly like all other queries.
    # PatentAgent is only used as a last-resort fallback at Step 4 when both KB and
    # web search return no context at all. See last-resort block below.

    # Step 3: KB retrieval + academic tools (research queries) started concurrently.
    #
    # For research queries we know we'll need academic papers if KB returns nothing,
    # so we fire the academic tool gather as an asyncio.Task at the same time as KB
    # retrieval — saving ~200 ms of sequential wait time.
    # If KB already has a paper for this query, we cancel the academic task immediately.
    academic_task: asyncio.Task | None = None
    research_query: str = ""
    if openalex_enabled and is_research_query:
        research_query = build_academic_query(cleaned_query, intent)
        logger.info(
            "orchestrator.research.query_cleaned",
            cleaned_query_preview=cleaned_query[:80],
            research_query=research_query,
            intent_entities=intent.get("entities"),
        )

        async def _run_academic_tools() -> tuple:
            # Use cleaned query directly - no query_policy transformation needed
            return await asyncio.gather(
                search_semantic_scholar(
                    primary_project_id,
                    research_query,
                    must_match_terms=intent.get("must_match_terms", []),
                    domain_terms=intent.get("domain_terms", []),
                    query_specificity=intent.get("query_specificity", "broad"),
                ),
                search_arxiv(
                    primary_project_id,
                    research_query,
                    must_match_terms=intent.get("must_match_terms", []),
                    domain_terms=intent.get("domain_terms", []),
                    query_specificity=intent.get("query_specificity", "broad"),
                ),
                search_pubmed(
                    primary_project_id,
                    research_query,
                    must_match_terms=intent.get("must_match_terms", []),
                    domain_terms=intent.get("domain_terms", []),
                    query_specificity=intent.get("query_specificity", "broad"),
                ),
                return_exceptions=True,
            )

        academic_task = asyncio.create_task(_run_academic_tools())
        logger.info(
            "orchestrator.research.academic_tools_started_concurrently",
            query_preview=research_query[:80],
        )

    # Skip KB if user explicitly requested web search (e.g. "search web for X", "google X")
    if force_web_search:
        logger.info("orchestrator.kb_skipped_explicit_web_search", query_preview=cleaned_query[:80])
        kb_results = None
    else:
        try:
            kb_results = await retrieve(
                query=cleaned_query,
                project_ids=relevant_project_ids,
                top_k=settings.KB_RETRIEVAL_TOP_K,
                exclude_papers=exclude_papers,
            )
        except Exception:
            # Cancel academic_task before propagating — otherwise the pending Task is
            # garbage-collected and Python logs "Task was destroyed but it is pending!".
            if academic_task is not None:
                academic_task.cancel()
                academic_task = None
            logger.exception("orchestrator.kb_retrieve_error", query_preview=cleaned_query[:80])
            raise

    # When the user names a specific site (target_domain is set), the intent is
    # "fetch live content from that site" — KB content is irrelevant regardless of
    # its cosine score.  Force kb_results = None so the search fallback always fires
    # with include_domains, guaranteeing specific page URLs in the synthesis context.
    if target_domain and kb_results is not None:
        logger.info(
            "orchestrator.kb_bypassed_for_target_domain",
            target_domain=target_domain,
            kb_chunk_count=len(kb_results),
        )
        kb_results = None

    # For specific queries, use intent-based entity matching to force fallback when KB lexical coverage is weak
    # even if cosine similarity crossed threshold.
    is_specific_query = intent.get("query_specificity") == "specific"
    must_match_terms = intent.get("must_match_terms", [])
    if kb_results is not None and is_specific_query and must_match_terms and not target_domain:
        kb_coverage = _max_coverage_for_kb(kb_results, must_match_terms)
        if kb_coverage < settings.QUERY_ENTITY_MATCH_THRESHOLD:
            logger.info(
                "orchestrator.kb_entity_coverage_below_threshold",
                coverage=round(kb_coverage, 3),
                threshold=settings.QUERY_ENTITY_MATCH_THRESHOLD,
                must_match_terms=must_match_terms,
            )
            kb_results = None

    kb_has_paper = any(chunk.get("source") == "paper" for chunk in (kb_results or []))
    openalex_papers: list[dict] = []

    # If KB already has a paper, cancel academic task — its results are not needed
    if kb_has_paper and academic_task is not None:
        academic_task.cancel()
        academic_task = None
        logger.info("orchestrator.research.academic_task_cancelled_kb_has_paper")

    # Step 3a: KB fallback — parallel search tool calls when KB score < 0.70
    # Task 2.2 improvement: 24.8s → 8.5s (16.3s savings)
    search_fallback_results: str = ""
    skip_general_fallback = bool(openalex_enabled and is_research_query)
    if kb_results is None and not skip_general_fallback:
        logger.info(
            "kb_score_below_threshold_triggering_fallback", query_preview=cleaned_query[:80]
        )
        try:
            # Call all 3 search tools in parallel with fault tolerance
            search_results = await asyncio.gather(
                search_tavily(
                    primary_project_id,
                    cleaned_query,
                    include_domains=include_domains,
                ),
                search_exa(
                    primary_project_id,
                    cleaned_query,
                    include_domains=include_domains,
                ),
                search_perigon(
                    primary_project_id,
                    cleaned_query,
                    must_match_terms=intent.get("must_match_terms", []),
                    domain_terms=intent.get("domain_terms", []),
                    query_specificity=intent.get("query_specificity", "broad"),
                ),
                return_exceptions=True,
            )

            # Collect valid results (handle individual tool failures gracefully)
            all_search_results: list[dict] = []
            tool_names = ["tavily", "exa", "perigon"]
            for idx, tool_result in enumerate(search_results):
                tool_name = tool_names[idx]
                if isinstance(tool_result, Exception):
                    logger.warning(
                        f"orchestrator.search_fallback.{tool_name}_error",
                        error=str(tool_result),
                    )
                    continue
                if isinstance(tool_result, list):
                    all_search_results.extend(tool_result)
                    logger.info(
                        f"orchestrator.search_fallback.{tool_name}_success",
                        result_count=len(tool_result),
                    )

            # Deduplicate by URL (exact match)
            seen_urls: set[str] = set()
            deduplicated_results: list[dict] = []
            for search_result in all_search_results:
                url = str(search_result.get("url") or "").strip().lower()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated_results.append(search_result)

            # Format results into context string
            if deduplicated_results:
                search_fallback_results = _format_search_fallback_context(deduplicated_results)
                logger.info(
                    "orchestrator.search_fallback_success",
                    total_results=len(all_search_results),
                    deduplicated_results=len(deduplicated_results),
                    result_length=len(search_fallback_results),
                )
            else:
                logger.warning("orchestrator.search_fallback_no_results")
        except Exception:
            logger.exception("orchestrator.search_fallback_error")
            search_fallback_results = ""
    elif kb_results is None and skip_general_fallback:
        logger.info(
            "orchestrator.search_fallback_skipped_for_research_query",
            query_preview=cleaned_query[:80],
        )

    # Step 3b: Research fallback — parallel tool calls with Exa fallback strategy (Phase 3)
    # academic_task was started concurrently with KB retrieval above (when is_research_query=True).
    # It is None if kb_has_paper was True (task was cancelled) or query is not a research query.
    research_papers: list[dict] = []
    if openalex_enabled and is_research_query and not kb_has_paper and academic_task is not None:
        stage_start = time.perf_counter()

        try:
            # Await the pre-started academic task (already running since KB retrieval started)
            logger.info(
                "orchestrator.research.academic_tools_await", query_preview=research_query[:80]
            )

            # academic_task returns a tuple from asyncio.gather with return_exceptions=True
            academic_results = await academic_task
            academic_task = None  # Consumed

            # Extract results (tools return [] on failure, never raise)
            semantic_scholar_papers = (
                academic_results[0] if isinstance(academic_results[0], list) else []
            )
            arxiv_papers = academic_results[1] if isinstance(academic_results[1], list) else []
            pubmed_papers = academic_results[2] if isinstance(academic_results[2], list) else []

            # Track tool calls for metrics
            if semantic_scholar_papers:
                research_tool_calls_total.labels(tool="semantic_scholar").inc()
            if arxiv_papers:
                research_tool_calls_total.labels(tool="arxiv").inc()
            if pubmed_papers:
                research_tool_calls_total.labels(tool="pubmed").inc()

            # Count total academic results
            academic_paper_count = (
                len(semantic_scholar_papers) + len(arxiv_papers) + len(pubmed_papers)
            )

            logger.info(
                "orchestrator.research.academic_tools_complete",
                semantic_scholar=len(semantic_scholar_papers),
                arxiv=len(arxiv_papers),
                pubmed=len(pubmed_papers),
                total=academic_paper_count,
            )

            # Phase 3.1: Exa fallback — only call if academic results < threshold
            exa_papers: list[dict] = []
            exa_called = False

            if academic_paper_count < settings.EXA_FALLBACK_THRESHOLD:
                logger.info(
                    "orchestrator.research.exa_fallback_triggered",
                    academic_count=academic_paper_count,
                    threshold=settings.EXA_FALLBACK_THRESHOLD,
                )
                try:
                    exa_papers = await search_exa(primary_project_id, research_query)
                    if exa_papers:
                        research_tool_calls_total.labels(tool="exa").inc()
                        exa_called = True
                    logger.info("orchestrator.research.exa_complete", result_count=len(exa_papers))
                except Exception:
                    logger.exception("orchestrator.research.exa_error")
                    exa_papers = []
            else:
                logger.info(
                    "orchestrator.research.exa_skipped",
                    academic_count=academic_paper_count,
                    threshold=settings.EXA_FALLBACK_THRESHOLD,
                )

            # Track Exa usage rate (0.0 if skipped, 1.0 if called)
            exa_usage_rate.observe(1.0 if exa_called else 0.0)

            # Phase 3.2: Deduplicate papers across all sources
            raw_paper_count = academic_paper_count + len(exa_papers)

            if raw_paper_count > 0:
                research_papers = deduplicate_papers(
                    semantic_scholar=semantic_scholar_papers,
                    arxiv=arxiv_papers,
                    pubmed=pubmed_papers,
                    exa=exa_papers,
                )

                # Normalize schemas across all sources so the formatter receives
                # consistent fields (publication_year, cited_by_count, doi, paper_id)
                # regardless of which tool produced each paper.
                research_papers = normalize_paper_schema(research_papers)
                if is_specific_query:
                    research_papers = _filter_papers_by_entity_coverage(
                        research_papers,
                        must_match_terms,
                    )

                # Track deduplication metrics
                deduped_count = len(research_papers)
                duplicates_removed = raw_paper_count - deduped_count
                duplicate_rate = (
                    duplicates_removed / raw_paper_count if raw_paper_count > 0 else 0.0
                )

                research_deduplication_rate.observe(duplicate_rate)

                logger.info(
                    "orchestrator.research.deduplication_complete",
                    raw_count=raw_paper_count,
                    deduped_count=deduped_count,
                    duplicates_removed=duplicates_removed,
                    duplicate_rate=round(duplicate_rate, 3),
                )
            else:
                logger.info("orchestrator.research.no_papers", query_preview=research_query[:80])

        except Exception:
            logger.exception("orchestrator.research.unexpected_error")
            research_papers = []
        finally:
            # Track research stage duration
            stage_duration = time.perf_counter() - stage_start
            orchestrator_stage_duration_seconds.labels(stage="research").observe(stage_duration)
            logger.info(
                "orchestrator.research.complete",
                paper_count=len(research_papers),
                duration_seconds=round(stage_duration, 2),
            )

    # Use research_papers instead of openalex_papers for the rest of the pipeline
    openalex_papers = research_papers

    # Fast path for explicit paper-list requests: return deterministic list so
    # users consistently get multiple papers instead of sparse summarisation.
    if openalex_papers and _asks_for_paper_list(cleaned_query) and not kb_has_paper:
        direct_answer = _format_openalex_paper_list_answer(cleaned_query, openalex_papers)
        yield f"data: {orjson.dumps({'token': direct_answer}).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"
        logger.info(
            "orchestrator.openalex_direct_answer",
            paper_count=len(openalex_papers),
            returned=min(10, len(openalex_papers)),
        )
        return

    # Step 4: Build context for synthesis
    social_insights = _extract_social_insights(kb_results or [])
    is_social_like_query = intent.get("query_type") in {"social", "trend"}

    # Calculate context quality metrics once for logging and prompt
    chunk_count = len(kb_results) if kb_results else 0
    avg_score = 0.0
    if kb_results:
        scores = [float(c.get("score", 0.0) or 0.0) for c in kb_results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        # Count source types for logging
        from collections import Counter
        source_counts: Counter = Counter()
        # Check for URL availability in paper metadata
        papers_with_doi = 0
        papers_with_url = 0
        papers_without_any_url = 0

        for chunk in kb_results:
            source = chunk.get("source", "unknown")
            source_counts[source] += 1

            # Track URL availability for papers
            if source == "paper":
                metadata = chunk.get("metadata") or {}
                doi = str(metadata.get("doi") or "").strip()
                url = str(metadata.get("url") or "").strip()
                paper_id = str(metadata.get("paper_id") or "").strip()
                if doi:
                    papers_with_doi += 1
                elif url:
                    papers_with_url += 1
                elif paper_id:
                    papers_with_url += 1  # paper_id can be used as fallback URL
                else:
                    papers_without_any_url += 1

        logger.info(
            "orchestrator.kb_hit",
            chunk_count=chunk_count,
            avg_score=round(avg_score, 3),
            min_score=round(min_score, 3),
            max_score=round(max_score, 3),
            source_counts=source_counts,
            papers_with_doi=papers_with_doi,
            papers_with_url=papers_with_url,
            papers_without_any_url=papers_without_any_url,
        )

    if kb_results:
        context = _format_kb_context(kb_results)
        if openalex_papers:
            context = f"{context}\n\n---\n\n{_format_openalex_context(openalex_papers)}"
    elif search_fallback_results:
        # KB returned None (score < threshold) — use search fallback results.
        # Each result already has its own [Title](url) header from _format_search_fallback_context;
        # no outer wrapper label needed.
        context = search_fallback_results
        logger.info(
            "orchestrator.search_fallback_context_used", result_length=len(search_fallback_results)
        )
    elif openalex_papers:
        context = _format_openalex_context(openalex_papers)
        logger.info("orchestrator.openalex_only_context", paper_count=len(openalex_papers))
    else:
        context = ""
        logger.info("orchestrator.kb_miss_or_below_threshold")

    # Step 4b: Last-resort PatentAgent — only for patent queries when KB and web search
    # both returned no context. This keeps the happy path (KB → web search) unchanged
    # and only pays the PatentsView API cost when we genuinely have no other evidence.
    #
    # PatentAgent output is fed into SynthesisAgent as context (not yielded directly)
    # so citation formatting, inline links, and answer structure are consistent with
    # all other query types. The patent tool already produces google_patents_url per result.
    patent_context = ""
    if not context and query_type == "patent" and not target_domain:
        logger.info("orchestrator.patent_agent_last_resort", query=cleaned_query[:80])
        try:
            patent_agent = _get_patent_agent()
            patent_result = await patent_agent.arun(cleaned_query)
            patent_content = (
                patent_result.content if patent_result and patent_result.content else ""
            )
            if patent_content:
                # Wrap patent output as context so SynthesisAgent can cite it inline
                patent_context = f"[Source: Patent Search Results]\n{patent_content}"
                context = patent_context
                logger.info(
                    "orchestrator.patent_agent_last_resort_success",
                    result_length=len(patent_content),
                )
            else:
                logger.warning("orchestrator.patent_agent_last_resort_empty")
        except Exception:
            logger.exception("orchestrator.patent_agent_last_resort_error")
            # Fall through to synthesis with empty context

    # Step 5: Build synthesis prompt
    if context:
        social_block = ""
        if is_social_like_query or social_insights:
            social_block = (
                "Structured Social Insights (precomputed from retrieved social evidence):\n"
                f"{orjson.dumps(social_insights).decode('utf-8')}\n\n"
            )

        # Build requirements dynamically — only instruct the LLM to include a section
        # when the underlying data actually exists, preventing hallucinated empty sections.
        req_parts = [
            "1) Write a comprehensive, multi-paragraph report that directly addresses the query. "
            "Extract ALL specific data from the context: numbers, percentages, names, dates, "
            "ratings, quotes, clinical findings, patent numbers, market figures. "
            "Do NOT reduce any source to a one-sentence summary — include all substantive detail it contains.",
        ]
        req_num = 2
        req_parts.append(
            f"{req_num}) CRITICAL — CITATIONS: Every factual claim MUST be cited inline using "
            "[title](url) markdown format. Use only URLs that appear verbatim in the context — "
            "never invent or guess a URL. If no URL is available, use [Source: title]."
        )
        req_num += 1
        if openalex_papers:
            req_parts.append(
                f"{req_num}) Include a '## Latest Papers' section with 6-8 distinct papers. "
                "For each paper include: title as a markdown link using its DOI or OpenAlex URL, "
                "publication year, and a 1-2 sentence summary of its key finding."
            )
            req_num += 1
        if is_social_like_query or social_insights:
            req_parts.append(
                f"{req_num}) Include a '## Social Insights' section grounded strictly in the "
                "provided metrics — do not add general social media commentary."
            )
            req_num += 1
        req_parts.append(
            f"{req_num}) Cite at least 3-4 distinct sources spread throughout the response, not "
            "clustered only at the end."
        )
        req_num += 1
        req_parts.append(
            f"{req_num}) NO AI BLOAT: Do not write generic introductory paragraphs, "
            "obvious background filler, or fluffy transitional sentences. "
            "Every sentence must contain a specific sourced fact. Cut anything that does not."
        )
        if is_specific_query and must_match_terms:
            req_num += 1
            req_parts.append(
                f"{req_num}) Prioritize evidence that explicitly mentions: "
                + ", ".join(must_match_terms)
                + "."
            )

        synthesis_prompt = (
            f"Research Query: {cleaned_query}\n\n"
            f"Domain: {intent.get('domain', 'general')}\n\n"
            + (
                f"Context Quality: You have {chunk_count} retrieved knowledge chunks with average "
                f"relevance score {avg_score:.2f} (pre-filtered above threshold "
                f">{settings.KB_SCORE_THRESHOLD}).\n\n"
                if chunk_count > 0
                else ""
            )
            + f"{social_block}"
            f"Relevant Context:\n{context}\n\n"
            f"Response requirements:\n" + "\n".join(req_parts)
        )
    else:
        synthesis_prompt = (
            f"Research Query: {cleaned_query}\n\n"
            f"Domain: {intent.get('domain', 'general')}\n\n"
            "No relevant documents were found in the knowledge base for this query. "
            "Provide a short, cautious summary that is still relevant to the query domain. "
            "Clearly state that KB evidence is currently insufficient."
        )

    # Step 6: Stream synthesis tokens
    synthesis_agent = _get_synthesis_agent()
    full_response_parts: list[str] = []

    try:
        async for event in synthesis_agent.arun(synthesis_prompt, stream=True, stream_events=True):
            if isinstance(event, RunContentEvent):
                delta = event.content
                if delta and isinstance(delta, str):
                    full_response_parts.append(delta)
                    yield f"data: {orjson.dumps({'token': delta}).decode('utf-8')}\n\n"
    except Exception:
        logger.exception("orchestrator.synthesis_error")
        error_msg = "I encountered an error generating the response. Please try again."
        yield f"data: {orjson.dumps({'token': error_msg}).decode('utf-8')}\n\n"

    yield "data: [DONE]\n\n"

    logger.info(
        "orchestrator.done",
        response_length=sum(len(p) for p in full_response_parts),
        kb_used=bool(kb_results),
    )
