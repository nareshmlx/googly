"""Extract text from fetched fulltext source assets."""

from __future__ import annotations

import io

import pypdf

from app.core.config import settings
from app.services.fulltext_types import FulltextExtractResult


def extract_text_from_asset(content_bytes: bytes, mime_type: str | None) -> FulltextExtractResult:
    """Extract text from asset bytes with PDF-first support for v1."""
    content_type = str(mime_type or "").lower()
    if "pdf" not in content_type:
        return FulltextExtractResult(status="unsupported", error_code="unsupported_mime")

    try:
        reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        page_count = min(len(reader.pages), int(settings.FULLTEXT_MAX_PAGES))
        parts: list[str] = []
        for idx in range(page_count):
            parts.append((reader.pages[idx].extract_text() or "").strip())
        text = "\n\n".join(part for part in parts if part).strip()
        if not text:
            return FulltextExtractResult(status="empty", page_count=page_count)
        return FulltextExtractResult(
            status="success",
            text=text,
            page_count=page_count,
            extracted_chars=len(text),
        )
    except Exception as exc:
        return FulltextExtractResult(
            status="failed",
            error_code="extract_failed",
            error_message=str(exc),
        )
