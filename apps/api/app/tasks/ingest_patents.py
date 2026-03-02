"""Patent source ingestion tasks (PatentsView, Lens, Web Fallback)."""

import structlog

from app.kb.ingester import RawDocument
from app.tasks.ingest_utils import (
    _looks_like_patent_url,
    _query_for_patents,
    _query_variants_for_source,
)
from app.tools.patents_lens import search_lens
from app.tools.patents_patentsview import search_patentsview
from app.tools.search_exa import search_exa
from app.tools.search_tavily import search_tavily

logger = structlog.get_logger(__name__)


async def _ingest_patents_patentsview(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch patents from PatentsView."""
    _ = redis
    query = _query_for_patents(
        intent,
        source="patentsview",
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_patentsview.no_query", project_id=project_id)
        return []

    try:
        patents: list[dict] = []
        seen_patents: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_patentsview(
                project_id,
                query_variant,
                must_match_terms=intent.get("must_match_terms") or [],
                domain_terms=intent.get("domain_terms") or [],
                query_specificity=intent.get("query_specificity"),
            )
            for patent in batch:
                patent_key = str(
                    patent.get("patent_number") or patent.get("url") or patent.get("title") or ""
                ).strip()
                if not patent_key or patent_key.lower() in seen_patents:
                    continue
                seen_patents.add(patent_key.lower())
                patents.append(patent)
    except Exception:
        logger.exception("ingest_patentsview.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for patent in patents:
        abstract = str(patent.get("content") or "").strip()
        if not abstract:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="patent",
                source_id=str(patent.get("patent_number") or patent.get("url") or ""),
                title=str(patent.get("title") or ""),
                content=abstract,  # Store full abstract (chunker handles chunking)
                metadata={
                    "patent_number": patent.get("patent_number") or "",
                    "date": patent.get("date") or "",
                    "inventors": patent.get("inventors") or [],
                    "url": patent.get("url") or "",
                    "fulltext_url": patent.get("url") or "",
                    "claims_snippet": abstract[:800],  # Increased from 400 to capture more context
                    "tool": "patentsview",
                },
            )
        )
    return docs


async def _ingest_patents_lens(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch patents from Lens."""
    _ = redis
    query = _query_for_patents(
        intent,
        source="lens",
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_lens.no_query", project_id=project_id)
        return []

    try:
        patents: list[dict] = []
        seen_patents: set[str] = set()
        for query_variant in _query_variants_for_source(intent, query):
            batch = await search_lens(project_id, query_variant)
            for patent in batch:
                patent_key = str(
                    patent.get("lens_id")
                    or patent.get("patent_number")
                    or patent.get("title")
                    or ""
                ).strip()
                if not patent_key or patent_key.lower() in seen_patents:
                    continue
                seen_patents.add(patent_key.lower())
                patents.append(patent)
    except Exception:
        logger.exception("ingest_lens.failed", project_id=project_id)
        return []

    docs: list[RawDocument] = []
    for patent in patents:
        abstract = str(patent.get("content") or "").strip()
        if not abstract:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="patent",
                source_id=str(patent.get("lens_id") or patent.get("patent_number") or ""),
                title=str(patent.get("title") or ""),
                content=abstract,  # Store full abstract (chunker handles chunking)
                metadata={
                    "lens_id": patent.get("lens_id") or "",
                    "patent_number": patent.get("patent_number") or "",
                    "jurisdiction": patent.get("jurisdiction") or "",
                    "kind": patent.get("kind") or "",
                    "date": patent.get("date") or "",
                    "inventors": patent.get("inventors") or [],
                    "url": patent.get("url") or "",
                    "fulltext_url": patent.get("url") or "",
                    "claims_snippet": abstract[:800],  # Increased from 400 to capture more context
                    "tool": "lens",
                },
            )
        )
    return docs


async def _ingest_patents_web_fallback(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
) -> list[RawDocument]:
    """Fallback patent ingest via web search."""
    query = _query_for_patents(
        intent,
        source="patent_web_fallback",
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        return []

    web_query = f"{query} patent"
    candidates: list[dict] = []
    try:
        candidates.extend(await search_exa(project_id, web_query))
    except Exception:
        logger.exception("patents_web_fallback.exa_failed", project_id=project_id)
    try:
        candidates.extend(await search_tavily(project_id, web_query))
    except Exception:
        logger.exception("patents_web_fallback.tavily_failed", project_id=project_id)

    docs: list[RawDocument] = []
    seen_urls: set[str] = set()
    for row in candidates:
        url = str(row.get("url") or "").strip()
        if not url or not _looks_like_patent_url(url) or url in seen_urls:
            continue
        seen_urls.add(url)
        content = (row.get("content") or row.get("snippet") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="patent",
                source_id=url,
                title=str(row.get("title") or "Patent result"),
                content=content,  # Store full content (chunker handles chunking)
                metadata={
                    "url": url,
                    "tool": "web_patent_fallback",
                    "fallback_origin": True,
                },
            )
        )
    return docs
