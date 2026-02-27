"""Typed contracts for fulltext enrichment pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FulltextResolveResult:
    status: str
    resolved_url: str | None = None
    canonical_url: str | None = None
    confidence: float = 0.0
    reason: str = ""
    source_fetcher: str = ""


@dataclass
class FulltextExtractResult:
    status: str
    text: str = ""
    page_count: int = 0
    extracted_chars: int = 0
    error_code: str = ""
    error_message: str = ""
