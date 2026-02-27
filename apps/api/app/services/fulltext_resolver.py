"""Resolve and canonicalize fulltext candidate URLs for paper/patent documents."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import structlog

from app.core.config import settings
from app.core.url_safety import is_safe_public_url
from app.kb.ingester import RawDocument
from app.services.fulltext_types import FulltextResolveResult

logger = structlog.get_logger(__name__)

_PAPER_OA_HOST_HINTS = {
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "doi.org",
    "openalex.org",
}


def canonicalize_url(url: str) -> str:
    """Normalize URL for deduplication by stripping fragment and tracking params."""
    parsed = urlparse(str(url or "").strip())
    clean_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if not k.lower().startswith("utm_")
    ]
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        fragment="",
        query=urlencode(clean_query),
    )
    return urlunparse(normalized)


def _allowed_domains() -> set[str]:
    raw = str(settings.FULLTEXT_ALLOWED_DOMAINS or "").strip()
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _paper_candidates(metadata: dict) -> list[tuple[str, str]]:
    """Return candidate URLs with provenance labels for paper fulltext resolution."""
    return [
        (str(metadata.get("open_access_url") or "").strip(), "open_access_url"),
        (str(metadata.get("pdf_url") or "").strip(), "pdf_url"),
        (str(metadata.get("url") or "").strip(), "url"),
        (str(metadata.get("doi") or "").strip(), "doi"),
    ]


def _normalize_candidate_url(candidate: str, fetcher: str) -> str:
    """Normalize non-URL DOI candidates into resolvable HTTPS URLs."""
    value = str(candidate or "").strip()
    if not value:
        return ""
    if fetcher == "doi" and not value.lower().startswith(("http://", "https://")):
        return f"https://doi.org/{value}"
    return value


def _patent_candidates(metadata: dict) -> list[tuple[str, str]]:
    """Return candidate URLs with provenance labels for patent fulltext resolution."""
    return [
        (str(metadata.get("fulltext_url") or "").strip(), "fulltext_url"),
        (str(metadata.get("pdf_url") or "").strip(), "pdf_url"),
        (str(metadata.get("url") or "").strip(), "url"),
    ]


def _paper_oa_allowed(url: str, metadata: dict) -> bool:
    """Return True when the paper URL looks open-access enough for v1 ingestion."""
    if bool(metadata.get("is_open_access") or metadata.get("open_access")):
        return True
    hostname = (urlparse(url).hostname or "").lower()
    if any(hostname == domain or hostname.endswith(f".{domain}") for domain in _PAPER_OA_HOST_HINTS):
        return True
    return "/pdf" in url.lower() or "pmc" in url.lower()


def resolve_fulltext_url(doc: RawDocument) -> FulltextResolveResult:
    """Resolve fulltext candidate URL for one paper/patent metadata document."""
    metadata = dict(doc.metadata or {})
    source = str(doc.source or "").strip().lower()
    allowed_domains = _allowed_domains()
    if source not in {"paper", "patent"}:
        return FulltextResolveResult(status="skipped", reason="unsupported_source")

    candidates = _paper_candidates(metadata) if source == "paper" else _patent_candidates(metadata)
    for candidate, fetcher in candidates:
        normalized_candidate = _normalize_candidate_url(candidate, fetcher)
        if not normalized_candidate:
            continue
        canonical = canonicalize_url(normalized_candidate)
        allowed, reason = is_safe_public_url(canonical, allowed_domains=allowed_domains or None)
        if not allowed:
            logger.info(
                "fulltext_resolve.blocked",
                source=source,
                reason=reason,
                candidate=normalized_candidate[:120],
            )
            continue
        if source == "paper" and not _paper_oa_allowed(canonical, metadata):
            continue
        return FulltextResolveResult(
            status="success",
            resolved_url=canonical,
            canonical_url=canonical,
            confidence=0.9,
            reason="resolved",
            source_fetcher=fetcher,
        )

    return FulltextResolveResult(
        status="blocked",
        confidence=0.0,
        reason="no_eligible_url",
    )
