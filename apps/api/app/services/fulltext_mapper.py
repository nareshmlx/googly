"""Map extracted fulltext payloads into KB ingester RawDocument objects."""

from __future__ import annotations

from app.kb.ingester import RawDocument


def map_fulltext_raw_document(
    *,
    project_id: str,
    user_id: str,
    source: str,
    source_id: str,
    title: str,
    text: str,
    base_metadata: dict,
    asset_id: str,
    source_fetcher: str,
    page_count: int,
) -> RawDocument:
    """Build a fulltext RawDocument preserving source metadata and idempotent source_id."""
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "content_level": "fulltext",
            "asset_id": asset_id,
            "section": "document",
            "page_start": 1 if page_count > 0 else None,
            "page_end": page_count if page_count > 0 else None,
            "source_fetcher": source_fetcher,
        }
    )
    return RawDocument(
        project_id=project_id,
        user_id=user_id,
        source=source,
        source_id=f"{source_id}:fulltext",
        title=title,
        content=text,
        metadata=metadata,
    )
