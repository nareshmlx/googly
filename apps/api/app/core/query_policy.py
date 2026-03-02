"""Query policy utilities for intent-aware, source-specific search construction.

This module centralizes query understanding so ingestion and runtime search both
use the same deterministic policy:
- Preserve user-mentioned specific entities as must-match terms.
- Expand optional context terms from intent without diluting specificity.
- Build source-specific query strings for papers, patents, and news tools.
- Provide lexical coverage scoring for post-retrieval relevance gating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_QUERY_STOPWORDS: set[str] = {
    "a",
    "about",
    "across",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "can",
    "find",
    "for",
    "from",
    "get",
    "give",
    "how",
    "in",
    "into",
    "is",
    "latest",
    "list",
    "me",
    "new",
    "newest",
    "of",
    "on",
    "or",
    "papers",
    "please",
    "recent",
    "research",
    "show",
    "studies",
    "study",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}

_META_TERMS: set[str] = {
    "paper",
    "papers",
    "research",
    "research paper",
    "research papers",
    "study",
    "studies",
    "latest",
    "recent",
    "newest",
    "current",
}

_DOMAIN_CONTEXT_TERMS: dict[str, list[str]] = {
    "beauty_market_intelligence": ["skincare", "cosmetics", "dermatology"],
    "cosmetics": ["skincare", "cosmetics", "dermatology"],
    "skincare": ["skincare", "dermatology", "topical"],
    "fragrance": ["fragrance", "cosmetic", "consumer"],
    "haircare": ["haircare", "scalp", "cosmetic"],
}


@dataclass(frozen=True)
class QueryPolicy:
    """Normalized query policy used by tools, reranking, and fallback logic."""

    original_query: str
    normalized_query: str
    must_match_terms: list[str]
    optional_terms: list[str]
    domain_terms: list[str]
    query_type: str
    query_specificity: str

    @property
    def is_specific(self) -> bool:
        """Return True when the query should preserve exact user entities."""
        return self.query_specificity == "specific" and bool(self.must_match_terms)


def _tokenize(text: str) -> list[str]:
    """Tokenize into lowercase alphanumeric tokens."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    """Deduplicate strings case-insensitively preserving first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = re.sub(r"\s+", " ", str(value or "").strip())
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_phrase(term: str) -> str:
    """Normalize phrase spacing and trim punctuation-like boundary chars."""
    cleaned = re.sub(r"\s+", " ", str(term or "").strip())
    cleaned = cleaned.strip(" ,;:.-_")
    return cleaned


def _entity_candidates_from_query(query: str) -> list[str]:
    """Extract deterministic entity-like candidates from raw query."""
    quoted = [m.group(1) for m in re.finditer(r'"([^"]+)"', query or "")]
    quoted = [_normalize_phrase(value) for value in quoted]

    raw_tokens = _tokenize(query)
    compact = [tok for tok in raw_tokens if tok not in _QUERY_STOPWORDS and len(tok) >= 3]
    # Use all token candidates — avoid flooding must-match with broad language.
    token_candidates = compact
    return _dedupe_keep_order(quoted + token_candidates)


def _specificity_from_query(query: str, intent: dict, explicit_specific_signals: bool) -> str:
    """Infer specificity if the intent payload does not provide one."""
    intent_specificity = str(intent.get("query_specificity") or "").strip().lower()
    if intent_specificity in {"specific", "broad"}:
        return intent_specificity

    if explicit_specific_signals:
        return "specific"

    query_tokens = [t for t in _tokenize(query) if t not in _QUERY_STOPWORDS]
    if len(query_tokens) <= 4 and query_tokens:
        return "specific"
    return "broad"


def lexical_entity_coverage(text: str, must_match_terms: list[str]) -> float:
    """Return the fraction of must-match terms present in the given text."""
    terms = [
        _normalize_phrase(term).lower() for term in must_match_terms if _normalize_phrase(term)
    ]
    if not terms:
        return 1.0
    haystack = re.sub(r"\s+", " ", str(text or "").lower())
    matched = 0
    for term in terms:
        if len(term) >= 3 and term in haystack:
            matched += 1
    return matched / len(terms)
