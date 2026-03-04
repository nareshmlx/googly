"""Shared query sanitization helpers for tool entrypoints."""

from __future__ import annotations

from app.core.config import settings


def sanitize_query(
    query: str,
    *,
    logger,
    empty_event: str,
    too_long_event: str,
    lower: bool = False,
) -> str | None:
    """Normalize, validate, and truncate a query string for external search tools."""
    cleaned = str(query or "").strip()
    if lower:
        cleaned = cleaned.lower()
    if not cleaned:
        logger.warning(empty_event)
        return None
    if len(cleaned) > settings.SEARCH_QUERY_MAX_LEN:
        logger.warning(too_long_event, original_length=len(cleaned))
        cleaned = cleaned[: settings.SEARCH_QUERY_MAX_LEN]
    return cleaned
