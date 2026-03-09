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

        # Use a generator expression for potentially better memory efficiency and
        # specify extraction_mode for improved handling of ambiguous Unicode characters.
        # Filter out empty parts directly in the generator.
        text_parts_generator = (
            (reader.pages[idx].extract_text(extraction_mode="layout") or "").strip()
            for idx in range(page_count)
        )

        # Join non-empty parts.
        text = "\n\n".join(part for part in text_parts_generator if part).strip()

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
