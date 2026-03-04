"""Paper source ingestion tasks (OpenAlex, Semantic Scholar, PubMed, arXiv)."""

import asyncio
from collections.abc import Awaitable, Callable

import structlog

from app.kb.ingester import RawDocument
from app.tasks.ingest_utils import (
    _query_for_papers,
    _query_variants_for_source,
    generate_stable_source_id,
)
from app.tools.papers_arxiv import search_arxiv
from app.tools.papers_openalex import fetch_papers
from app.tools.papers_pubmed import search_pubmed
from app.tools.papers_semantic_scholar import search_semantic_scholar

logger = structlog.get_logger(__name__)

SearchFn = Callable[[str, str], Awaitable[list[dict]]]
PaperKeyFn = Callable[[dict], str]
PaperContentFn = Callable[[dict], str]
PaperMetadataFn = Callable[[dict], dict]


async def _ingest_papers_source(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    source_key: str,
    social_filter: str,
    project_title: str,
    project_description: str,
    search_fn: SearchFn,
    error_event: str,
    empty_query_event: str | None,
    key_fn: PaperKeyFn,
    content_fn: PaperContentFn,
    metadata_fn: PaperMetadataFn,
) -> list[RawDocument]:
    """Generic paper-source ingestion flow with pluggable mapping functions."""
    query = _query_for_papers(
        intent,
        source=source_key,
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        if empty_query_event:
            logger.warning(empty_query_event, project_id=project_id)
        return []

    papers: list[dict] = []
    seen_papers: set[str] = set()
    variants = list(_query_variants_for_source(intent, query))
    tasks = [search_fn(project_id, qv) for qv in variants]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for qv, batch in zip(variants, results, strict=False):
        if isinstance(batch, asyncio.CancelledError):
            raise
        if isinstance(batch, Exception):
            logger.exception(error_event, project_id=project_id, query=qv)
            continue
        if isinstance(batch, BaseException):
            raise batch
        for paper in batch:
            paper_key = key_fn(paper).strip()
            if not paper_key:
                continue
            normalized_key = paper_key.lower()
            if normalized_key in seen_papers:
                continue
            seen_papers.add(normalized_key)
            papers.append(paper)

    docs: list[RawDocument] = []
    for paper in papers:
        content = content_fn(paper).strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=generate_stable_source_id(paper),
                title=paper.get("title", ""),
                content=content,
                metadata=metadata_fn(paper),
            )
        )
    return docs


async def _ingest_papers_openalex(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    project_title: str = "",
    project_description: str = "",
) -> list[RawDocument]:
    """Fetch research papers from OpenAlex."""
    return await _ingest_papers_source(
        project_id=project_id,
        user_id=user_id,
        intent=intent,
        source_key="openalex",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
        search_fn=fetch_papers,
        error_event="ingest_openalex.failed_batch",
        empty_query_event="ingest_openalex.no_query",
        key_fn=lambda paper: str(
            paper.get("paper_id") or paper.get("doi") or paper.get("title") or ""
        ),
        content_fn=lambda paper: str(paper.get("abstract") or ""),
        metadata_fn=lambda paper: {
            "doi": paper.get("doi", ""),
            "publication_year": paper.get("publication_year", 0),
            "url": paper.get("url") or "",
            "pdf_url": paper.get("pdf_url") or "",
            "open_access_url": paper.get("open_access_url") or "",
            "is_open_access": bool(paper.get("is_open_access") or False),
            # paper_id stored so _format_kb_context can construct a clickable link
            # when doi and url are both empty.
            "paper_id": paper.get("paper_id", ""),
            "tool": "openalex",
        },
    )


async def _ingest_papers_semantic_scholar(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch papers from Semantic Scholar."""
    _ = redis
    return await _ingest_papers_source(
        project_id=project_id,
        user_id=user_id,
        intent=intent,
        source_key="semantic_scholar",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
        search_fn=search_semantic_scholar,
        error_event="ingest_semantic_scholar.failed",
        empty_query_event=None,
        key_fn=lambda paper: str(
            paper.get("paper_id") or paper.get("id") or paper.get("title") or ""
        ),
        content_fn=lambda paper: str(paper.get("abstract") or ""),
        metadata_fn=lambda paper: {
            "doi": paper.get("doi", ""),
            "publication_year": paper.get("year", 0),
            "url": paper.get("url") or "",
            # paper_id stored so _format_kb_context can construct a clickable link
            # when doi and url are both empty (e.g. papers with no open-access DOI).
            "paper_id": paper.get("paper_id", ""),
            "tool": "semantic_scholar",
        },
    )


async def _ingest_papers_pubmed(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch papers from PubMed."""
    _ = redis
    return await _ingest_papers_source(
        project_id=project_id,
        user_id=user_id,
        intent=intent,
        source_key="pubmed",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
        search_fn=search_pubmed,
        error_event="ingest_pubmed.failed",
        empty_query_event=None,
        key_fn=lambda paper: str(paper.get("pmid") or paper.get("title") or ""),
        content_fn=lambda paper: str(paper.get("abstract") or ""),
        metadata_fn=lambda paper: {
            "pmid": paper.get("pmid", ""),
            "doi": paper.get("doi", ""),
            "publication_year": paper.get("publication_year", 0),
            "url": paper.get("url") or "",
            "tool": "pubmed",
        },
    )


async def _ingest_papers_arxiv(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str = "",
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch papers from arXiv."""
    _ = redis
    return await _ingest_papers_source(
        project_id=project_id,
        user_id=user_id,
        intent=intent,
        source_key="arxiv",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
        search_fn=search_arxiv,
        error_event="ingest_arxiv.failed",
        empty_query_event=None,
        key_fn=lambda paper: str(paper.get("arxiv_id") or paper.get("title") or ""),
        content_fn=lambda paper: str(paper.get("summary") or ""),
        metadata_fn=lambda paper: {
            "arxiv_id": paper.get("arxiv_id", ""),
            "doi": paper.get("doi", ""),
            "publication_year": paper.get("publication_year") or paper.get("year", 0),
            "url": paper.get("url") or "",
            "pdf_url": paper.get("pdf_url") or "",
            "tool": "arxiv",
        },
    )
