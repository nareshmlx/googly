"""Shared query sanitization helpers for tool entrypoints."""

from __future__ import annotations

import re
from typing import Any

from app.core.config import settings

_LABELED_LINE_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9 /()_-]{1,80}:\s*")
_SIGNAL_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
_QUERY_NOISE_TERMS: set[str] = {
    "a",
    "an",
    "and",
    "am",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "cause",
    "caused",
    "causes",
    "causing",
    "could",
    "did",
    "do",
    "does",
    "for",
    "get",
    "getting",
    "give",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "know",
    "list",
    "me",
    "my",
    "project",
    "projects",
    "goal",
    "goals",
    "research",
    "conduct",
    "conducting",
    "focusing",
    "focused",
    "focus",
    "user",
    "users",
    "pain",
    "points",
    "primary",
    "stakeholder",
    "stakeholders",
    "key",
    "decisions",
    "decision",
    "output",
    "format",
    "scope",
    "evidence",
    "priorities",
    "constraints",
    "assumptions",
    "success",
    "signals",
    "next",
    "steps",
    "short",
    "term",
    "latest",
    "recent",
    "need",
    "needs",
    "help",
    "of",
    "off",
    "on",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "those",
    "to",
    "us",
    "want",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
    "or",
}
_LOW_SIGNAL_SINGLE_TOKENS: set[str] = {
    "brand",
    "brands",
    "drop",
    "product",
    "products",
    "theme",
    "themes",
}
_PHRASE_FOLLOWER_TOKENS: set[str] = {"off"}
_SHORT_SIGNAL_TOKEN_RE = re.compile(r"[a-z0-9]{1,2}")



def strip_labeled_text(text: str) -> str:
    """Remove leading label prefixes from structured brief text while preserving content."""
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = _LABELED_LINE_RE.sub("", raw_line).strip()
        if line:
            lines.append(line)
    if lines:
        return " ".join(lines).strip()
    return str(text or "").strip()


def _is_signal_token(token: str) -> bool:
    """Return True when a token can stand alone as topical search signal."""
    return (
        len(token) >= 3
        and not token.isdigit()
        and token not in _QUERY_NOISE_TERMS
        and token not in _LOW_SIGNAL_SINGLE_TOKENS
    )


def _is_contextual_short_token(token: str) -> bool:
    """Allow short suffix tokens only when attached to a stronger leading term."""
    return len(token) <= 2 and bool(_SHORT_SIGNAL_TOKEN_RE.fullmatch(token))


def _ordered_signal_phrases(cleaned: str) -> tuple[list[str], set[str]]:
    """Extract protected signal phrases in source order before single-token fallback."""
    matches = list(_SIGNAL_TOKEN_RE.finditer(cleaned))
    phrases: list[str] = []
    protected_tokens: set[str] = set()
    seen: set[str] = set()

    for index, match in enumerate(matches):
        raw = match.group(0).lower()

        if "-" in raw:
            parts = [part.strip() for part in raw.split("-") if part.strip()]
            if (
                len(parts) >= 2
                and any(_is_signal_token(part) for part in parts)
                and all(len(part) >= 2 and not part.isdigit() for part in parts)
            ):
                phrase = " ".join(parts).strip()
                if phrase and phrase not in seen:
                    seen.add(phrase)
                    phrases.append(phrase)
                    protected_tokens.update(parts)

        if index >= len(matches) - 1:
            continue

        token = raw
        next_token = matches[index + 1].group(0).lower()
        if not _is_signal_token(token):
            continue
        if not _is_contextual_short_token(next_token):
            continue

        phrase = f"{token} {next_token}".strip()
        if phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        protected_tokens.update((token, next_token))

    return phrases, protected_tokens


def _normalize_candidate_phrase(text: str, *, max_words: int = 3) -> str:
    """Convert free text into a compact signal phrase, dropping control/noise words."""
    cleaned = strip_labeled_text(text).lower().replace("_", " ").replace("-", " ")
    raw_tokens = [token for token in _SIGNAL_TOKEN_RE.findall(cleaned) if token]
    if raw_tokens:
        explicit_tokens: list[str] = []
        if (
            len(raw_tokens) >= 2
            and len(raw_tokens) <= max_words
            and len(raw_tokens[0]) >= 2
            and not raw_tokens[0].isdigit()
            and raw_tokens[0] not in _QUERY_NOISE_TERMS
            and any(_is_signal_token(token) for token in raw_tokens[1:])
        ):
            explicit_tokens.append(raw_tokens[0])
        for token in raw_tokens:
            normalized = str(token or "").strip().lower()
            if not normalized or normalized.isdigit():
                continue
            if explicit_tokens and normalized == explicit_tokens[0]:
                continue
            if _is_signal_token(normalized):
                explicit_tokens.append(normalized)
                continue
            if (
                explicit_tokens
                and len(raw_tokens) <= max_words
                and (
                    _is_contextual_short_token(normalized)
                    or normalized in _PHRASE_FOLLOWER_TOKENS
                )
            ):
                explicit_tokens.append(normalized)
        if len(explicit_tokens) >= 2:
            return " ".join(explicit_tokens[:max_words]).strip()

    words = extract_signal_terms(text, max_terms=max_words)
    if not words:
        return ""
    flattened: list[str] = []
    seen: set[str] = set()
    for item in words:
        for token in item.split():
            if token in seen:
                continue
            seen.add(token)
            flattened.append(token)
            if len(flattened) >= max_words:
                return " ".join(flattened).strip()
    return " ".join(flattened).strip()


def extract_signal_terms(text: str, *, max_terms: int = 8) -> list[str]:
    """Extract bounded, de-duplicated topical terms from noisy project/search prose."""
    cleaned = strip_labeled_text(text).lower()
    out: list[str] = []
    seen: set[str] = set()
    phrases, protected_tokens = _ordered_signal_phrases(cleaned)
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= max_terms:
            return out

    for match in _SIGNAL_TOKEN_RE.finditer(cleaned):
        raw = match.group(0)
        parts = raw.split("-")
        for part in parts:
            token = str(part or "").strip().lower()
            if token in protected_tokens:
                continue
            if not _is_signal_token(token):
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= max_terms:
                return out
    return out


def derive_must_match_terms(
    *,
    explicit_terms: list[str] | tuple[str, ...] | None = None,
    entities: list[str] | tuple[str, ...] | None = None,
    keywords: list[str] | tuple[str, ...] | None = None,
    domain_terms: list[str] | tuple[str, ...] | None = None,
    description: str = "",
    min_terms: int = 3,
    max_terms: int = 6,
) -> list[str]:
    """Derive bounded topical anchors from structured intent fields without hardcoding domains."""
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        phrase = _normalize_candidate_phrase(raw)
        if not phrase:
            return
        if phrase in seen:
            return
        seen.add(phrase)
        out.append(phrase)

    for collection in (explicit_terms or [], entities or [], keywords or [], domain_terms or []):
        for value in collection:
            _push(str(value or ""))
            if len(out) >= max_terms:
                return out

    if len(out) < min_terms and description:
        for term in extract_signal_terms(description, max_terms=max_terms * 2):
            _push(term)
            if len(out) >= max_terms:
                break

    return out[:max_terms]


def sanitize_search_seed(text: str, *, max_terms: int = 8) -> str:
    """Collapse noisy brief text into a concise search seed suitable for source queries."""
    terms = extract_signal_terms(text, max_terms=max_terms)
    if terms:
        return " ".join(terms).strip()
    return re.sub(r"\s+", " ", strip_labeled_text(text)).strip()


def sanitize_query(
    query: str,
    *,
    logger: Any,
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
