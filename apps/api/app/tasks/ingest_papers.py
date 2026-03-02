"""Paper source ingestion tasks (OpenAlex, Semantic Scholar, PubMed, arXiv)."""

import asyncio
import structlog

from app.kb.ingester import RawDocument
from app.tasks.ingest_utils import _query_for_papers, _query_variants_for_source
from app.tools.papers_arxiv import search_arxiv
from app.tools.papers_openalex import fetch_papers
from app.tools.papers_pubmed import search_pubmed
from app.tools.papers_semantic_scholar import search_semantic_scholar

logger = structlog.get_logger(__name__)


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
    query = _query_for_papers(
        intent,
        source="openalex",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        logger.warning("ingest_openalex.no_query", project_id=project_id)
        return []

    papers: list[dict] = []
    seen_papers: set[str] = set()
    variants = list(_query_variants_for_source(intent, query))

    tasks = [fetch_papers(project_id, qv) for qv in variants]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for qv, batch in zip(variants, results):
        if isinstance(batch, Exception):
            logger.exception("ingest_openalex.failed_batch", project_id=project_id, query=qv)
            continue
        for paper in batch:
            paper_key = str(
                paper.get("paper_id") or paper.get("doi") or paper.get("title") or ""
            ).strip()
            if not paper_key:
                continue
            normalized_key = paper_key.lower()
            if normalized_key in seen_papers:
                continue
            seen_papers.add(normalized_key)
            papers.append(paper)

    docs: list[RawDocument] = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=paper.get("paper_id", ""),
                title=paper.get("title", ""),
                content=abstract,
                metadata={
                    "doi": paper.get("doi", ""),
                    "publication_year": paper.get("publication_year", 0),
                    "url": paper.get("url") or "",
                    "pdf_url": paper.get("pdf_url") or "",
                    "open_access_url": paper.get("open_access_url") or "",
                    "is_open_access": bool(paper.get("is_open_access") or False),
                    "tool": "openalex",
                },
            )
        )
    return docs


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
    query = _query_for_papers(
        intent,
        source="semantic_scholar",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        return []

    papers: list[dict] = []
    seen_papers: set[str] = set()
    variants = list(_query_variants_for_source(intent, query))

    tasks = [search_semantic_scholar(project_id, qv) for qv in variants]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for qv, batch in zip(variants, results):
        if isinstance(batch, Exception):
            logger.exception("ingest_semantic_scholar.failed", project_id=project_id, query=qv)
            continue
        for paper in batch:
            paper_key = str(
                paper.get("paper_id") or paper.get("id") or paper.get("title") or ""
            ).strip()
            if not paper_key or paper_key.lower() in seen_papers:
                continue
            seen_papers.add(paper_key.lower())
            papers.append(paper)

    docs: list[RawDocument] = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=paper.get("paper_id") or paper.get("id", ""),
                title=paper.get("title", ""),
                content=abstract,
                metadata={
                    "doi": paper.get("doi", ""),
                    "publication_year": paper.get("year", 0),
                    "url": paper.get("url") or "",
                    "tool": "semantic_scholar",
                },
            )
        )
    return docs


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
    query = _query_for_papers(
        intent,
        source="pubmed",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        return []

    papers: list[dict] = []
    seen_papers: set[str] = set()
    variants = list(_query_variants_for_source(intent, query))

    tasks = [search_pubmed(project_id, qv) for qv in variants]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for qv, batch in zip(variants, results):
        if isinstance(batch, Exception):
            logger.exception("ingest_pubmed.failed", project_id=project_id, query=qv)
            continue
        for paper in batch:
            paper_key = str(paper.get("pmid") or paper.get("title") or "").strip()
            if not paper_key or paper_key.lower() in seen_papers:
                continue
            seen_papers.add(paper_key.lower())
            papers.append(paper)

    docs: list[RawDocument] = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=paper.get("pmid", ""),
                title=paper.get("title", ""),
                content=abstract,
                metadata={
                    "pmid": paper.get("pmid", ""),
                    "doi": paper.get("doi", ""),
                    "publication_year": paper.get("publication_year", 0),
                    "url": paper.get("url") or "",
                    "tool": "pubmed",
                },
            )
        )
    return docs


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
    query = _query_for_papers(
        intent,
        source="arxiv",
        social_filter=social_filter,
        project_title=project_title,
        project_description=project_description,
    )
    if not query:
        return []

    papers: list[dict] = []
    seen_papers: set[str] = set()
    variants = list(_query_variants_for_source(intent, query))

    tasks = [search_arxiv(project_id, qv) for qv in variants]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for qv, batch in zip(variants, results):
        if isinstance(batch, Exception):
            logger.exception("ingest_arxiv.failed", project_id=project_id, query=qv)
            continue
        for paper in batch:
            paper_key = str(paper.get("arxiv_id") or paper.get("title") or "").strip()
            if not paper_key or paper_key.lower() in seen_papers:
                continue
            seen_papers.add(paper_key.lower())
            papers.append(paper)

    docs: list[RawDocument] = []
    for paper in papers:
        summary = (paper.get("summary") or "").strip()
        if not summary:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=paper.get("arxiv_id", ""),
                title=paper.get("title", ""),
                content=summary,
                metadata={
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "doi": paper.get("doi", ""),
                    "publication_year": paper.get("publication_year") or paper.get("year", 0),
                    "url": paper.get("url") or "",
                    "pdf_url": paper.get("pdf_url") or "",
                    "tool": "arxiv",
                },
            )
        )
    return docs
