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
import json
import math
import re
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from agno.run.agent import RunContentEvent

from app.agents.intent import build_intent_agent
from app.agents.patent import build_patent_agent
from app.agents.project_relevance import build_project_relevance_agent
from app.agents.research import build_research_agent
from app.agents.search_fallback import build_search_fallback_agent
from app.agents.synthesis import build_synthesis_agent
from app.agents.trend import build_trend_agent
from app.core.config import settings
from app.core.dedup_papers import deduplicate_papers
from app.core.metrics import (
    exa_usage_rate,
    orchestrator_stage_duration_seconds,
    research_deduplication_rate,
    research_tool_calls_total,
)
from app.core.paper_schema import normalize_paper_schema
from app.kb.retriever import retrieve
from app.tools.news_perigon import search_perigon
from app.tools.papers_arxiv import search_arxiv
from app.tools.papers_pubmed import search_pubmed
from app.tools.papers_semantic_scholar import search_semantic_scholar
from app.tools.search_exa import search_exa
from app.tools.search_tavily import search_tavily

logger = structlog.get_logger(__name__)

# Module-level singletons — initialised once, reused across all requests
# (Agno agents are stateless per arun() call — safe to share across requests)
_intent_agent = None
_synthesis_agent = None
_project_relevance_agent = None
_search_fallback_agent = None
_research_agent = None
_trend_agent = None
_patent_agent = None


def _get_intent_agent():
    global _intent_agent
    if _intent_agent is None:
        _intent_agent = build_intent_agent()
    return _intent_agent


def _get_synthesis_agent():
    global _synthesis_agent
    if _synthesis_agent is None:
        _synthesis_agent = build_synthesis_agent()
    return _synthesis_agent


def _get_project_relevance_agent():
    global _project_relevance_agent
    if _project_relevance_agent is None:
        _project_relevance_agent = build_project_relevance_agent()
    return _project_relevance_agent


def _get_search_fallback_agent():
    global _search_fallback_agent
    if _search_fallback_agent is None:
        _search_fallback_agent = build_search_fallback_agent()
    return _search_fallback_agent


def _get_research_agent():
    global _research_agent
    if _research_agent is None:
        _research_agent = build_research_agent()
    return _research_agent


def _get_trend_agent():
    global _trend_agent
    if _trend_agent is None:
        _trend_agent = build_trend_agent()
    return _trend_agent


def _get_patent_agent():
    global _patent_agent
    if _patent_agent is None:
        _patent_agent = build_patent_agent()
    return _patent_agent


def _safe_int(value: object) -> int:
    """Convert mixed numeric values (int/float/str) to int safely."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return 0
        try:
            return int(float(cleaned))
        except ValueError:
            return 0
    return 0


def _extract_social_insights(kb_results: list[dict]) -> dict:
    """
    Build compact social evidence summary from retrieved KB chunks.

    This gives synthesis a grounded structure instead of raw caption-only context.
    """
    social_chunks = [
        c for c in kb_results if c.get("source") in {"social_instagram", "social_tiktok"}
    ]
    if not social_chunks:
        return {}

    platform_counts: dict[str, int] = {"instagram": 0, "tiktok": 0}
    creator_counts: dict[str, int] = {}
    top_posts: list[dict] = []
    newest_ts: int | None = None
    oldest_ts: int | None = None

    for chunk in social_chunks:
        metadata = chunk.get("metadata") or {}
        source = chunk.get("source")
        platform = "instagram" if source == "social_instagram" else "tiktok"
        platform_counts[platform] += 1

        author = (
            metadata.get("username")
            or metadata.get("author_username")
            or (chunk.get("title") or "").lstrip("@")
            or "unknown"
        )
        author = str(author)
        creator_counts[author] = creator_counts.get(author, 0) + 1

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

    top_creator_pairs = sorted(creator_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    top_creators = [{"author": author, "posts": posts} for author, posts in top_creator_pairs]

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
    """Format OpenAlex papers into synthesis context blocks."""
    parts: list[str] = []
    for paper in papers:
        title = str(paper.get("title") or "Untitled paper")
        paper_id = str(paper.get("paper_id") or "")
        year = paper.get("publication_year") or "n/a"
        # Try both field names — legacy OpenAlex uses `abstract`, tools use `content`
        abstract = str(paper.get("abstract") or paper.get("content") or "").strip()
        summary = abstract[:1200] if abstract else "Abstract unavailable."
        doi = str(paper.get("doi") or "")
        source_label = paper_id or title
        meta_line = f"publication_year={year}"
        if doi:
            meta_line += f" doi={doi}"
        parts.append(f"[Source: {source_label} (paper)]\ntitle={title}\n{meta_line}\n{summary}")
    return "\n\n---\n\n".join(parts)


def _clean_openalex_query(query: str, intent: dict) -> str:
    """Build a focused keyword query for academic search APIs from intent or raw query.

    Academic APIs (arXiv, PubMed, Semantic Scholar) do keyword matching, not
    natural-language understanding.  Sending a full English question sentence like
    "What are the latest research papers on peptide anti-aging?" causes false-positive
    matches on stop-words ("research", "papers", "latest") returning completely
    off-topic results.

    Strategy (in priority order):
    1. Use intent["entities"] when available — the IntentAgent already extracted the
       key terms, so "peptide anti-aging" is far cleaner than the full sentence.
    2. Fall back to stripping question words and common filler from the raw query.
    3. Append domain-specific boost terms when the domain is beauty/cosmetics and
       no beauty keywords are present in the query.
    """
    intent = intent or {}

    # ── 1. Use intent entities (primary path) ────────────────────────────────
    # Strip meta-query terms the IntentAgent sometimes includes as entities
    # (e.g. "research papers", "latest studies") — these are query type descriptors,
    # not subject keywords, and will match off-topic academic papers.
    _META_TERMS = frozenset(
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
    entities: list[str] = [
        e.strip()
        for e in (intent.get("entities") or [])
        if isinstance(e, str) and e.strip() and e.strip().lower() not in _META_TERMS
    ]
    if entities:
        text = " ".join(entities[:5])
        # Add domain boost if domain is beauty-adjacent and not already in entities
        lowered = text.lower()
        domain = str(intent.get("domain") or "").lower()
        if domain in {"cosmetics", "beauty_market_intelligence", "fragrance", "skincare"}:
            if not any(t in lowered for t in {"beauty", "cosmetics", "skincare", "fragrance"}):
                text = f"{text} skincare cosmetics"
        return re.sub(r"\s+", " ", text).strip()

    # ── 2. Fallback: strip question words and filler from raw query ───────────
    text = str(query or "")
    # Remove prompt-injection markers (prior assistant output leaked in)
    for marker in ("\n\nAnswer", "\nAnswer", "\n\nEvidence", "\nEvidence"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break

    # Remove question starters ("What are the", "Tell me about", etc.)
    text = re.sub(
        r"^(what|which|who|where|when|how|why|tell me|show me|find|list|give me)"
        r"\s+(are|is|do|does|were|was|the|me|a|an)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Remove research-query filler that matches off-topic papers via keyword overlap
    text = re.sub(
        r"\b(latest|recent|newest|current|research\s+papers?|papers?|studies|study"
        r"|about|regarding|on|for|from|in|a|the)\b\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 220:
        text = text[:220].rstrip()

    # Add domain boost if applicable and not already present
    lowered = text.lower()
    if not any(
        term in lowered for term in {"beauty", "cosmetics", "skincare", "makeup", "fragrance"}
    ):
        domain = str(intent.get("domain") or "").lower()
        if domain in {"cosmetics", "beauty_market_intelligence", "fragrance", "skincare"}:
            text = f"{text} skincare cosmetics".strip()

    return text


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

        # Truncate content to max 600 chars per result
        if len(content) > 600:
            content = f"{content[:600]}..."

        source_label = f"{title} ({source})" if title != "Untitled" else source
        parts.append(f"[Source: {source_label}]\nURL: {url}\n{content}")

    return "\n\n---\n\n".join(parts)


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
        link = doi or paper_id or url_fallback or "Link unavailable"
        citations = _safe_int(paper.get("cited_by_count"))
        context = _paper_context_snippet(paper)
        lines.append(f"{idx}. {title} ({year})")
        lines.append(f"   - {link}")
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


async def _extract_intent(query: str) -> dict:
    """
    Run IntentAgent and parse the JSON response.

    Returns a default intent dict on any failure — never raises — so a bad
    intent classification degrades gracefully instead of crashing the pipeline.
    """
    default = {
        "domain": "general",
        "query_type": "general",
        "entities": [],
        "confidence": 0.5,
        "is_research_query": False,
    }
    try:
        agent = _get_intent_agent()
        result = await agent.arun(query)
        text = result.content if result and result.content else ""
        if isinstance(text, str) and text.strip():
            parsed = json.loads(text.strip())
            return {**default, **parsed}
    except json.JSONDecodeError:
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

    projects_json = json.dumps(
        [
            {"id": p["id"], "title": p.get("title", ""), "description": p.get("description", "")}
            for p in all_projects
        ],
        indent=2,
    )
    prompt = f"query: {query}\n\nprojects:\n{projects_json}"

    try:
        agent = _get_project_relevance_agent()
        result = await agent.arun(prompt)
        text = result.content if result and result.content else ""
        if isinstance(text, str) and text.strip():
            parsed = json.loads(text.strip())
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
    except json.JSONDecodeError:
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
    intent, relevant_project_ids = await asyncio.gather(
        _extract_intent(cleaned_query),
        _get_relevant_project_ids(cleaned_query, primary_project_id, all_projects),
        return_exceptions=False,
    )

    query_type = intent.get("query_type", "general")
    is_research_query = bool(intent.get("is_research_query", False)) or _looks_like_research_query(
        cleaned_query
    )
    exclude_papers = not openalex_enabled or not is_research_query
    logger.info(
        "orchestrator.intent_and_relevance",
        domain=intent.get("domain"),
        query_type=query_type,
        entities=intent.get("entities"),
        is_research_query=is_research_query,
        exclude_papers=exclude_papers,
        relevant_project_ids=relevant_project_ids,
    )

    # Step 1a: Route to specialized agents based on intent
    if query_type == "trend":
        # Route to TrendAgent for trend analysis
        logger.info("orchestrator.routing_to_trend_agent", query=cleaned_query[:80])
        try:
            trend_agent = _get_trend_agent()
            trend_result = await trend_agent.arun(cleaned_query)
            trend_content = trend_result.content if trend_result and trend_result.content else ""

            if trend_content:
                # Stream the trend analysis result directly
                yield f"data: {json.dumps({'token': trend_content})}\n\n"
                yield "data: [DONE]\n\n"
                logger.info("orchestrator.trend_agent_success", result_length=len(trend_content))
                return
            else:
                logger.warning("orchestrator.trend_agent_empty_result")
                # Fall through to normal KB flow
        except Exception:
            logger.exception("orchestrator.trend_agent_error")
            # Fall through to normal KB flow on error

    elif query_type == "patent":
        # Route to PatentAgent for patent search
        logger.info("orchestrator.routing_to_patent_agent", query=cleaned_query[:80])
        try:
            patent_agent = _get_patent_agent()
            patent_result = await patent_agent.arun(cleaned_query)
            patent_content = (
                patent_result.content if patent_result and patent_result.content else ""
            )

            if patent_content:
                # Stream the patent search result directly
                yield f"data: {json.dumps({'token': patent_content})}\n\n"
                yield "data: [DONE]\n\n"
                logger.info("orchestrator.patent_agent_success", result_length=len(patent_content))
                return
            else:
                logger.warning("orchestrator.patent_agent_empty_result")
                # Fall through to normal KB flow
        except Exception:
            logger.exception("orchestrator.patent_agent_error")
            # Fall through to normal KB flow on error

    # Step 3: KB retrieval across all relevant projects
    # (relevant_project_ids already computed in parallel with intent in Step 1)
    kb_results = await retrieve(
        query=cleaned_query, project_ids=relevant_project_ids, exclude_papers=exclude_papers
    )

    kb_has_paper = any(chunk.get("source") == "paper" for chunk in (kb_results or []))
    openalex_papers: list[dict] = []

    # Step 3a: KB fallback — parallel search tool calls when KB score < 0.70
    # Task 2.2 improvement: 24.8s → 8.5s (16.3s savings)
    search_fallback_results: str = ""
    if kb_results is None:
        logger.info(
            "kb_score_below_threshold_triggering_fallback", query_preview=cleaned_query[:80]
        )
        try:
            # Call all 3 search tools in parallel with fault tolerance
            search_results = await asyncio.gather(
                search_tavily(cleaned_query),
                search_exa(cleaned_query),
                search_perigon(cleaned_query),
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

    # Step 3b: Research fallback — parallel tool calls with Exa fallback strategy (Phase 3)
    research_papers: list[dict] = []
    if openalex_enabled and is_research_query and not kb_has_paper:
        research_query = _clean_openalex_query(cleaned_query, intent)
        stage_start = time.perf_counter()
        logger.info(
            "orchestrator.research.query_cleaned",
            cleaned_query_preview=cleaned_query[:80],
            research_query=research_query,
            intent_entities=intent.get("entities"),
        )

        try:
            # Phase 3.1: Call academic tools first (Semantic Scholar, arXiv, PubMed) in parallel
            logger.info(
                "orchestrator.research.academic_tools_start", query_preview=research_query[:80]
            )

            # Call academic tools in parallel (return_exceptions=True to handle individual failures)
            academic_results = await asyncio.gather(
                search_semantic_scholar(research_query),
                search_arxiv(research_query),
                search_pubmed(research_query),
                return_exceptions=True,
            )

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
                    exa_papers = await search_exa(research_query)
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
        yield f"data: {json.dumps({'token': direct_answer})}\n\n"
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

    if kb_results:
        context_parts = []
        for chunk in kb_results:
            title = chunk.get("title") or "Untitled"
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0)
            metadata = chunk.get("metadata") or {}
            if source in {"social_instagram", "social_tiktok"}:
                likes = _safe_int(metadata.get("like_count") or metadata.get("likes"))
                views = _safe_int(metadata.get("view_count") or metadata.get("views"))
                author = (
                    metadata.get("username")
                    or metadata.get("author_username")
                    or str(title).lstrip("@")
                )
                context_parts.append(
                    f"[Source: {title} ({source}, relevance: {score:.2f})]\n"
                    f"author={author} likes={likes} views={views}\n"
                    f"{chunk['content']}"
                )
            else:
                context_parts.append(
                    f"[Source: {title} ({source}, relevance: {score:.2f})]\n{chunk['content']}"
                )
        if openalex_papers:
            context_parts.append(_format_openalex_context(openalex_papers))
        context = "\n\n---\n\n".join(context_parts)
        logger.info("orchestrator.kb_hit", chunk_count=len(kb_results))
    elif search_fallback_results:
        # KB returned None (score < threshold) — use search fallback results
        context = f"[Source: External Search Fallback]\n{search_fallback_results}"
        logger.info(
            "orchestrator.search_fallback_context_used", result_length=len(search_fallback_results)
        )
    elif openalex_papers:
        context = _format_openalex_context(openalex_papers)
        logger.info("orchestrator.openalex_only_context", paper_count=len(openalex_papers))
    else:
        context = ""
        logger.info("orchestrator.kb_miss_or_below_threshold")

    # Step 5: Build synthesis prompt
    if context:
        social_block = ""
        if is_social_like_query or social_insights:
            social_block = (
                "Structured Social Insights (precomputed from retrieved social evidence):\n"
                f"{json.dumps(social_insights, ensure_ascii=True)}\n\n"
            )
        synthesis_prompt = (
            f"Research Query: {cleaned_query}\n\n"
            f"Domain: {intent.get('domain', 'general')}\n\n"
            f"{social_block}"
            f"Relevant Knowledge Base Context:\n{context}\n\n"
            "Response requirements:\n"
            "1) Start with a concise answer tailored to the query.\n"
            "2) If paper sources are present, include a 'Latest Papers' section "
            "with 6-8 distinct papers when available.\n"
            "For each paper include: title, publication year, and DOI or OpenAlex link.\n"
            "3) If social evidence is present, include a 'Social Insights' section "
            "grounded in the provided metrics.\n"
            "4) Include an 'Evidence' section with specific sources used.\n"
            "5) If evidence is thin, include a brief 'Confidence / Gaps' note.\n"
            "Do not include irrelevant general information."
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
                    yield f"data: {json.dumps({'token': delta})}\n\n"
    except Exception:
        logger.exception("orchestrator.synthesis_error")
        error_msg = "I encountered an error generating the response. Please try again."
        yield f"data: {json.dumps({'token': error_msg})}\n\n"

    yield "data: [DONE]\n\n"

    logger.info(
        "orchestrator.done",
        response_length=sum(len(p) for p in full_response_parts),
        kb_used=bool(kb_results),
    )
