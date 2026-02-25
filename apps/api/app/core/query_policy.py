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
    # Preserve top terms only — avoid flooding must-match with broad language.
    token_candidates = compact[:3]
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


def build_query_policy(query: str, intent: dict | None = None) -> QueryPolicy:
    """Build a deterministic query policy from user query and structured intent."""
    intent = intent or {}
    normalized_query = re.sub(r"\s+", " ", str(query or "").strip())
    entities = intent.get("entities") or []
    explicit_must_terms = intent.get("must_match_terms") or []
    entity_terms = [
        _normalize_phrase(e)
        for e in entities
        if isinstance(e, str) and _normalize_phrase(e).lower() not in _META_TERMS
    ]
    must_match_terms = _dedupe_keep_order(
        entity_terms + [_normalize_phrase(e) for e in explicit_must_terms if isinstance(e, str)]
    )
    if not must_match_terms:
        must_match_terms = _entity_candidates_from_query(normalized_query)

    optional_terms = []
    for value in (intent.get("expanded_terms") or []):
        if isinstance(value, str):
            optional_terms.append(_normalize_phrase(value))
    for value in (intent.get("keywords") or []):
        if isinstance(value, str):
            optional_terms.append(_normalize_phrase(value))
    optional_terms = [
        term
        for term in _dedupe_keep_order(optional_terms)
        if term and term.lower() not in {v.lower() for v in must_match_terms}
    ][:8]

    domain = str(intent.get("domain") or "").strip().lower()
    intent_domain_terms = []
    for value in (intent.get("domain_terms") or []):
        if isinstance(value, str):
            intent_domain_terms.append(_normalize_phrase(value))
    intent_domain_terms = _dedupe_keep_order(intent_domain_terms)
    domain_terms = intent_domain_terms[:4] if intent_domain_terms else _DOMAIN_CONTEXT_TERMS.get(domain, [])[:3]
    query_type = str(intent.get("query_type") or "general").strip().lower()
    explicit_specific_signals = bool(entity_terms or explicit_must_terms or re.search(r'"[^"]+"', normalized_query))
    query_specificity = _specificity_from_query(normalized_query, intent, explicit_specific_signals)

    return QueryPolicy(
        original_query=str(query or ""),
        normalized_query=normalized_query,
        must_match_terms=must_match_terms[:5],
        optional_terms=optional_terms,
        domain_terms=domain_terms,
        query_type=query_type or "general",
        query_specificity=query_specificity,
    )


def build_source_query(policy: QueryPolicy, source: str) -> str:
    """Build source-specific query text from a normalized query policy."""
    source_key = str(source or "").strip().lower()
    must_terms = policy.must_match_terms[:3]
    optional = policy.optional_terms[:4]
    domain_terms = policy.domain_terms[:2]

    if source_key == "pubmed":
        must_clause = ""
        if must_terms:
            must_expr = " OR ".join(f'"{term}"[Title/Abstract]' for term in must_terms)
            must_clause = f"({must_expr})"
        else:
            must_clause = f'"{policy.normalized_query}"[Title/Abstract]'

        optional_all = optional + domain_terms
        if optional_all:
            optional_expr = " OR ".join(f'"{term}"[Title/Abstract]' for term in optional_all)
            return f"{must_clause} AND ({optional_expr})"
        return must_clause

    if source_key == "arxiv":
        if must_terms:
            pieces = [f'ti:"{term}" OR abs:"{term}"' for term in must_terms]
            core = " OR ".join(f"({piece})" for piece in pieces)
            if optional:
                optional_expr = " OR ".join(f'all:"{term}"' for term in optional[:2])
                return f"({core}) AND ({optional_expr})"
            return core
        return f'all:"{policy.normalized_query}"'

    if source_key in {"semantic_scholar", "openalex", "exa", "tavily", "perigon"}:
        if not policy.is_specific and policy.normalized_query:
            return policy.normalized_query
        pieces = [f'"{term}"' for term in must_terms] if policy.is_specific else must_terms
        pieces.extend(optional)
        pieces.extend(domain_terms)
        query = " ".join(piece for piece in pieces if piece).strip()
        return query or policy.normalized_query

    if source_key in {"patentsview", "lens"}:
        if not policy.is_specific and policy.normalized_query:
            return policy.normalized_query
        pieces = must_terms
        pieces.extend(optional[:3])
        pieces.extend(domain_terms)
        query = " ".join(piece for piece in pieces if piece).strip()
        return query or policy.normalized_query

    return policy.normalized_query


def lexical_entity_coverage(text: str, must_match_terms: list[str]) -> float:
    """Return the fraction of must-match terms present in the given text."""
    terms = [_normalize_phrase(term).lower() for term in must_match_terms if _normalize_phrase(term)]
    if not terms:
        return 1.0
    haystack = re.sub(r"\s+", " ", str(text or "").lower())
    matched = 0
    for term in terms:
        if len(term) >= 3 and term in haystack:
            matched += 1
    return matched / len(terms)
