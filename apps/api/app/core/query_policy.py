"""Lexical query policy utilities used by runtime relevance gating."""

from __future__ import annotations

import re


def _normalize_phrase(term: str) -> str:
    """Normalize phrase spacing and trim punctuation-like boundary chars."""
    cleaned = re.sub(r"\s+", " ", str(term or "").strip())
    cleaned = cleaned.strip(" ,;:.-_")
    return cleaned


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
