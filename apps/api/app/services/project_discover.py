"""Project discover-feed mapping helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

import structlog

from app.core.constants import SourceType
from app.core.utils import metadata_pick, parse_metadata
from app.models.schemas import DiscoverItem

logger = structlog.get_logger(__name__)


def row_to_discover_item(row: dict) -> DiscoverItem | None:
    """Map a DB row into the unified DiscoverItem response schema."""
    raw_source = str(row.get("source") or "search")
    source: Literal[
        "tiktok",
        "instagram",
        "youtube",
        "reddit",
        "x",
        "paper",
        "patent",
        "news",
        "search",
    ] | None = cast(
        Literal[
            "tiktok",
            "instagram",
            "youtube",
            "reddit",
            "x",
            "paper",
            "patent",
            "news",
            "search",
        ]
        | None,
        SourceType.DISCOVER_SOURCE_MAP.get(raw_source),
    )
    if raw_source in SourceType.DISCOVER_DIRECT_SOURCES:
        source = cast(
            Literal[
                "tiktok",
                "instagram",
                "youtube",
                "reddit",
                "x",
                "paper",
                "patent",
                "news",
                "search",
            ],
            raw_source,
        )
    if source is None:
        logger.warning("projects.discover.unsupported_source", source=raw_source)
        return None

    metadata = parse_metadata(row.get("metadata"))
    title = str(
        row.get("title")
        or metadata.get("title")
        or metadata.get("author")
        or metadata.get("byline")
        or "Untitled"
    )
    url_raw = metadata_pick(metadata, ("url", "doi"), row.get("url"))
    url = str(url_raw) if url_raw else None

    row_created_at = row.get("created_at")
    if isinstance(row_created_at, datetime):
        created_at_str = row_created_at.isoformat()
    elif row_created_at is not None:
        created_at_str = str(row_created_at)
    else:
        created_at_str = None

    published_raw = metadata_pick(
        metadata,
        ("published_at", "pubDate", "year"),
        row.get("published_at"),
        created_at_str,
    )
    published_at = str(published_raw) if published_raw else None

    summary_raw = row.get("summary") or row.get("content") or metadata.get("summary") or ""
    score_raw = row.get("score")
    try:
        score = float(score_raw) if score_raw is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0

    cover_raw = metadata.get("cover_url") or metadata.get("thumbnail_url")
    cover_url = str(cover_raw) if cover_raw else None
    author_raw = metadata.get("author") or metadata.get("byline")
    author = str(author_raw) if author_raw else None

    return DiscoverItem(
        source=source,
        item_id=str(row.get("item_id") or row.get("id") or row.get("source_id") or ""),
        title=title,
        summary=str(summary_raw)[:500],
        url=url,
        cover_url=cover_url,
        author=author,
        published_at=published_at,
        score=score,
        metadata=metadata,
    )
