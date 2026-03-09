"""Discovery source ingestion tasks (Perigon news, Tavily web, Exa web)."""

import asyncio

import structlog

from app.kb.ingester import RawDocument
from app.tasks.ingest_utils import _query_for_news_or_web, _query_variants_for_source
from app.tools.news_perigon import search_perigon
from app.tools.search_exa import search_exa
from app.tools.search_tavily import search_tavily

logger = structlog.get_logger(__name__)


async def _ingest_news(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch news from Perigon."""
    _ = redis
    query_intent = dict(intent or {})
    if project_title:
        query_intent.setdefault("project_title", project_title)
    if project_description:
        query_intent.setdefault("project_description", project_description)
    query = _query_for_news_or_web(query_intent, source="perigon")
    if not query:
        return []

    stories: list[dict] = []
    seen_ids: set[str] = set()
    variants = list(_query_variants_for_source(intent, query))

    tasks = [search_perigon(project_id, qv) for qv in variants]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for qv, batch in zip(variants, results, strict=False):
        if isinstance(batch, asyncio.CancelledError):
            raise
        if isinstance(batch, Exception):
            logger.exception("ingest_news.failed", project_id=project_id, query=qv)
            continue
        if isinstance(batch, BaseException):
            raise batch
        for story in batch:
            sid = str(story.get("article_id") or story.get("url") or "").strip()
            if not sid or sid.lower() in seen_ids:
                continue
            seen_ids.add(sid.lower())
            stories.append(story)

    docs: list[RawDocument] = []
    for story in stories:
        content = (story.get("content") or story.get("summary") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="news",
                source_id=str(story.get("article_id") or story.get("url") or ""),
                title=story.get("title", ""),
                content=content,
                metadata={
                    "url": story.get("url") or "",
                    "source_name": story.get("source_name") or "",
                    "published_at": story.get("published_at") or "",
                    "summary": story.get("content") or story.get("summary") or "",
                    "cover_url": (
                        story.get("cover_url")
                        or story.get("image_url")
                        or story.get("imageUrl")
                        or story.get("thumbnail_url")
                        or ""
                    ),
                    "tool": "perigon",
                },
            )
        )
    return docs


async def _ingest_web_tavily(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch web results from Tavily."""
    return await _ingest_web_source(
        project_id=project_id,
        user_id=user_id,
        intent=intent,
        project_title=project_title,
        project_description=project_description,
        source_key="tavily",
        tool_name="tavily",
        search_fn=search_tavily,
        error_event="ingest_web_tavily.failed",
    )


async def _ingest_web_exa(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    redis,
) -> list[RawDocument]:
    """Fetch web results from Exa."""
    return await _ingest_web_source(
        project_id=project_id,
        user_id=user_id,
        intent=intent,
        project_title=project_title,
        project_description=project_description,
        source_key="exa",
        tool_name="exa",
        search_fn=search_exa,
        error_event="ingest_web_exa.failed",
    )


async def _ingest_web_source(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    project_title: str = "",
    project_description: str = "",
    source_key: str,
    tool_name: str,
    search_fn,
    error_event: str,
) -> list[RawDocument]:
    """Fetch web results for one search provider and map to RawDocument."""
    query_intent = dict(intent or {})
    if project_title:
        query_intent.setdefault("project_title", project_title)
    if project_description:
        query_intent.setdefault("project_description", project_description)
    query = _query_for_news_or_web(query_intent, source=source_key)
    if not query:
        return []

    results: list[dict] = []
    seen_urls: set[str] = set()
    variants = list(_query_variants_for_source(intent, query))
    tasks = [search_fn(project_id, qv) for qv in variants]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    for qv, batch in zip(variants, batch_results, strict=False):
        if isinstance(batch, asyncio.CancelledError):
            raise
        if isinstance(batch, Exception):
            logger.exception(error_event, project_id=project_id, query=qv)
            continue
        if isinstance(batch, BaseException):
            raise batch
        for row in batch:
            url = str(row.get("url") or "").strip()
            if not url or url.lower() in seen_urls:
                continue
            seen_urls.add(url.lower())
            results.append(row)

    docs: list[RawDocument] = []
    for row in results:
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        content = (row.get("content") or row.get("snippet") or "").strip()
        if not content:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="search",
                source_id=url,
                title=row.get("title", ""),
                content=content,
                metadata={
                    "url": row.get("url", ""),
                    "summary": row.get("content") or row.get("snippet") or "",
                    "cover_url": (
                        row.get("cover_url")
                        or row.get("image_url")
                        or row.get("thumbnail_url")
                        or row.get("image")
                        or ""
                    ),
                    "tool": tool_name,
                },
            )
        )
    return docs
