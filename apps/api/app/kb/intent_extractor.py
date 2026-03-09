"""Intent extraction and refinement for project knowledge bases.

Two plain async functions — NOT Agno agents. These are structured LLM calls
(JSON in, JSON out) with no tool use. Using plain functions keeps the call path
simpler and avoids Agno agent overhead for what is essentially a transformation step.

extract_intent: called synchronously at project creation.
refine_intent:  called after ingest_document to detect domain shifts.
"""

import json
import re
from datetime import UTC, datetime
from typing import Any

import structlog

from app.core.config import settings
from app.core.openai_client import get_openai_client, openai_completions_with_circuit_breaker
from app.core.query_sanitize import (
    derive_must_match_terms,
    extract_signal_terms,
    sanitize_search_seed,
    strip_labeled_text,
)

logger = structlog.get_logger(__name__)


_EXTRACT_SYSTEM = """\
You are a domain-agnostic research intent analyst.
Extract structured research intent from a project description.
Output ONLY valid JSON — no prose, no markdown fences.

Schema:
{
  "domain": "<specific snake_case domain label, e.g. cosmetic_formulation, battery_recycling, llm_safety>",
  "entities": ["<exact named entities such as brands, products, ingredients, compounds, standards>"],
  "domain_terms": ["<supporting category, use-case, or technical context terms>"],
  "keywords": ["<ordered keywords with anchors first, then supporting context>"],
  "must_match_terms": ["<strongest anchors that must remain central to retrieval>"],
  "search_filters": {
    "news": "<natural language query for news APIs — domain terms/entities, no hashtags>",
    "papers": "<natural language query for academic paper APIs — technical/scientific terms, no hashtags, no # symbols>",
    "patents": "<natural language query for patent APIs — company names + technical terms, no hashtags>",
    "tiktok": "<hashtags for TikTok search when applicable; may be empty if topic is not social-native>",
    "social": "<same as tiktok - kept for backwards compatibility>",
    "instagram": "<1-3 word phrase for Instagram account search, no hashtags>"
  },
  "confidence": <float 0.0-1.0>
}

Rules:
- domain: most specific label possible, always snake_case - think like a librarian classifying a journal
- entities: preserve exact multi-word phrases when the user names a brand, product, ingredient, company, framework, or standard
- domain_terms: supporting context only; examples include category, use-case, problem, audience, or technical mechanism
- keywords: 3-10 ordered terms; put the strongest anchors first, then supporting context
- must_match_terms: 2-6 strongest anchors only; prefer exact named entities over generic descriptors
- Never force a domain that is not present in the user description or sample text.
- Never split a brand or product name into fragments when the full phrase is available.
- Avoid weak generic terms such as "brand", "product", "products", "themes", "research", "analysis", "latest".
- When brand + product are both present, include both in entities and must_match_terms.
- When ingredient + category are present, keep the ingredient first and the category/context terms after it.
- search_filters.papers: natural language only - no hashtags, no # symbols, no boolean operators
- search_filters.tiktok: hashtag format (#) when used; keep concise and topic-focused
- search_filters.social: exact copy of tiktok (backwards compatibility)
- search_filters.instagram: SHORT - 1-3 words max, descriptive, no hashtags
- confidence: 1.0 = unambiguous domain, 0.5 = could be multiple domains
- Output ONLY the JSON object, nothing else

Examples:
Input: "Track Vitamin C claim performance in skincare sunscreen and moisturizer products."
Output:
{
  "domain": "skincare_claim_analysis",
  "entities": ["Vitamin C"],
  "domain_terms": ["skincare", "sunscreen", "moisturizer", "claim performance"],
  "keywords": ["Vitamin C", "skincare", "sunscreen", "moisturizer", "claim performance"],
  "must_match_terms": ["Vitamin C", "sunscreen", "moisturizer"],
  "search_filters": {
    "news": "Vitamin C skincare sunscreen moisturizer claim performance",
    "papers": "Vitamin C skincare sunscreen moisturizer claim performance",
    "patents": "Vitamin C sunscreen moisturizer formulation claims",
    "tiktok": "#vitaminc #skincare #sunscreen #moisturizer",
    "social": "#vitaminc #skincare #sunscreen #moisturizer",
    "instagram": "vitamin c skincare"
  },
  "confidence": 0.89
}

Input: "Analyze La Roche-Posay Cicaplast Baume B5 and its barrier-repair positioning."
Output:
{
  "domain": "skincare_products",
  "entities": ["La Roche-Posay", "Cicaplast Baume B5"],
  "domain_terms": ["barrier repair", "sensitive skin", "balm"],
  "keywords": ["Cicaplast Baume B5", "La Roche-Posay", "barrier repair", "balm"],
  "must_match_terms": ["La Roche-Posay", "Cicaplast Baume B5"],
  "search_filters": {
    "news": "La Roche-Posay Cicaplast Baume B5 barrier repair",
    "papers": "Cicaplast Baume B5 barrier repair sensitive skin",
    "patents": "La Roche-Posay Cicaplast Baume B5 balm formulation",
    "tiktok": "#larocheposay #cicaplastbaumeb5 #barrierrepair",
    "social": "#larocheposay #cicaplastbaumeb5 #barrierrepair",
    "instagram": "laroche posay cicaplast"
  },
  "confidence": 0.94
}

Input: "Research ceramide ingredients in skincare products, focusing on anti-aging claims and clinical efficacy."
Output:
{
  "domain": "skincare_ingredients",
  "entities": ["ceramide"],
  "domain_terms": ["skincare", "anti-aging claims", "clinical efficacy", "ingredient mechanisms"],
  "keywords": ["ceramide", "skincare", "anti-aging claims", "clinical efficacy", "ingredient mechanisms"],
  "must_match_terms": ["ceramide", "clinical efficacy", "anti-aging claims"],
  "search_filters": {
    "news": "ceramide skincare anti-aging claims clinical efficacy",
    "papers": "ceramide skincare anti-aging claims clinical efficacy ingredient mechanisms",
    "patents": "ceramide skincare formulation clinical efficacy",
    "tiktok": "#ceramide #skincare #skinbarrier #antiaging",
    "social": "#ceramide #skincare #skinbarrier #antiaging",
    "instagram": "ceramide skincare"
  },
  "confidence": 0.9
}
"""

_REFINE_SYSTEM = """\
You refine an existing structured research intent based on new document content.
Output ONLY valid JSON — no prose, no markdown fences.

You will receive:
1. existing_intent: the current structured intent JSON
2. new_content_summaries: list of short summaries from newly uploaded document chunks

Rules:
- Only update the domain if the new content strongly suggests a different domain
  AND your confidence in the new domain is > 0.75
- Merge keywords additively - union of existing + new, deduplicated, max 12 total
- Update search_filters only if new keywords materially change the meaning of the search
- Preserve all existing search_filters fields including tiktok, social, and instagram
- search_filters.social must always equal search_filters.tiktok (backwards compatibility)
- search_filters.instagram must remain SHORT - 1-3 words max, no hashtags
- search_filters.papers must remain natural language only - no hashtags, no # symbols
- If nothing significant changed, return the existing_intent unchanged
- Output ONLY the JSON object, nothing else
"""

_GENERIC_KEYWORDS: set[str] = {
    "research",
    "trend",
    "trends",
    "product",
    "products",
    "market",
    "industry",
    "general",
    "analysis",
    "viral",
    "track",
    "breaking",
    "launch",
    "launches",
    "popular",
}

_STOPWORD_KEYWORDS: set[str] = {
    "and",
    "or",
    "for",
    "with",
    "from",
    "about",
    "using",
    "best",
    "top",
    "new",
}
_ANALYSIS_MODIFIER_KEYWORDS: set[str] = {
    "claim",
    "claims",
    "drop",
    "drop off",
    "decline",
    "declining",
    "performance",
    "risk",
    "risks",
    "theme",
    "themes",
}

_RELATED_KEYWORD_HINTS: dict[str, list[str]] = {
    "cosmetic_formulation": [
        "retinol",
        "niacinamide",
        "hyaluronic acid",
        "peptides",
        "ceramides",
        "aha",
        "bha",
        "stability testing",
        "in vitro efficacy",
        "skin barrier",
    ],
    "skincare": [
        "retinol",
        "niacinamide",
        "vitamin c",
        "ceramides",
        "hyaluronic acid",
        "peptides",
        "sunscreen",
        "sensitive skin",
    ],
    "haircare": [
        "keratin",
        "bond repair",
        "scalp barrier",
        "sulfate-free",
        "silicone-free",
        "protein treatment",
        "heat damage",
    ],
}


async def extract_intent(
    description: str,
    sample_text: str | None = None,
) -> dict:
    """
    Extract structured intent from a project description using the configured LLM.

    sample_text is optional — pass the first few hundred characters of an
    uploaded document to give the model more context at project creation time.

    Returns the parsed intent dict. On any LLM or parse failure, returns a
    safe default so project creation never fails due to intent extraction.
    """
    clean_description = strip_labeled_text(description)
    default_seed = sanitize_search_seed(clean_description, max_terms=8) or clean_description[:80]
    default_filters: dict[str, str] = {
        "news": default_seed,
        "papers": default_seed,
        "patents": default_seed,
        "tiktok": "",
        "social": "",
        "instagram": "",
    }
    default: dict[str, Any] = {
        "domain": "general",
        "entities": [],
        "domain_terms": [],
        "keywords": [],
        "must_match_terms": [],
        "search_filters": default_filters,
        "confidence": 0.0,
    }

    if not settings.OPENAI_API_KEY:
        logger.warning("intent_extractor.no_api_key")
        return _postprocess_intent(default, description=clean_description)

    prompt = f"Project description: {clean_description}"
    if sample_text:
        prompt += f"\n\nSample document text (first 500 chars):\n{sample_text[:500]}"

    logger.info("intent_extractor.extract.start", description_preview=description[:60])

    try:
        client = get_openai_client()
        create_comp = openai_completions_with_circuit_breaker(client)
        response = await create_comp(
            model=settings.INTENT_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=settings.INTENT_EXTRACT_TIMEOUT_SECONDS,
        )
        text = response.choices[0].message.content or ""
        intent = json.loads(text.strip())
        logger.info(
            "intent_extractor.extract.success",
            domain=intent.get("domain"),
            confidence=intent.get("confidence"),
        )
        # Deep-merge search_filters so missing keys fall back to defaults
        intent_filters = intent.get("search_filters")
        if isinstance(intent_filters, dict):
            safe_filters = {str(k): str(v) for k, v in intent_filters.items()}
        else:
            safe_filters = {}
        merged_filters = {**default_filters, **safe_filters}
        # social must always mirror tiktok — unconditional, never diverge.
        merged_filters["social"] = merged_filters.get("tiktok") or merged_filters.get("social", "")
        result = {**default, **intent, "search_filters": merged_filters}
        result = _postprocess_intent(result, description=clean_description)
        return result
    except json.JSONDecodeError:
        logger.warning(
            "intent_extractor.extract.parse_error",
            description_preview=description[:60],
        )
        return _postprocess_intent(default, description=clean_description)
    except Exception as exc:
        logger.error(
            "intent_extractor.extract.error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return _postprocess_intent(default, description=clean_description)


async def refine_intent(
    existing_intent: dict,
    new_chunk_summaries: list[str],
) -> dict:
    """
    Refine structured intent after new document chunks are ingested.

    Runs after ingest_document to detect domain shifts — e.g., a project
    originally about "vegan leather" receives a doc about "plant-based dyes",
    which might expand the keyword set without changing the core domain.

    Only updates if new content confidence > 0.75 — low-signal docs (e.g., a
    legal disclaimer or a table of contents) should not pollute the intent.

    Returns the refined intent dict with "refined_at" added. On any failure,
    returns the existing_intent unchanged so the KB is never left without intent.
    """
    if not settings.OPENAI_API_KEY or not new_chunk_summaries:
        return existing_intent

    summaries_text = "\n".join(f"- {s}" for s in new_chunk_summaries[:20])
    prompt = (
        f"existing_intent:\n{json.dumps(existing_intent, indent=2)}\n\n"
        f"new_content_summaries:\n{summaries_text}"
    )

    logger.info(
        "intent_extractor.refine.start",
        existing_domain=existing_intent.get("domain"),
        summary_count=len(new_chunk_summaries),
    )

    try:
        client = get_openai_client()
        create_comp = openai_completions_with_circuit_breaker(client)
        response = await create_comp(
            model=settings.INTENT_MODEL,
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=settings.INTENT_EXTRACT_TIMEOUT_SECONDS,
        )
        text = response.choices[0].message.content or ""
        refined = json.loads(text.strip())
        refined["refined_at"] = datetime.now(UTC).isoformat()
        logger.info(
            "intent_extractor.refine.success",
            new_domain=refined.get("domain"),
            keyword_count=len(refined.get("keywords", [])),
        )
        existing_filters = _as_mapping(existing_intent.get("search_filters"))
        refined_filters = _as_mapping(refined.get("search_filters"))
        merged_filters = {**existing_filters, **refined_filters}
        merged_filters["social"] = merged_filters.get("tiktok") or merged_filters.get("social", "")
        refined["search_filters"] = merged_filters
        return _postprocess_intent({**existing_intent, **refined}, description="")
    except json.JSONDecodeError:
        logger.warning("intent_extractor.refine.parse_error")
        return existing_intent
    except Exception as exc:
        logger.error(
            "intent_extractor.refine.error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return existing_intent


async def extract_document_intent(new_chunk_summaries: list[str]) -> dict:
    """Extract intent signal from uploaded document summaries only."""
    if not new_chunk_summaries:
        return {}
    combined = "\n".join(s for s in new_chunk_summaries if s.strip())
    if not combined.strip():
        return {}
    return await extract_intent(
        "Document-based intent refinement from uploaded project files.",
        sample_text=combined[:4000],
    )


def _weighted_union(description_terms: list[str], document_terms: list[str]) -> list[str]:
    """Merge terms with description-first weighting and deterministic ordering."""
    rank: dict[str, tuple[int, int, str]] = {}
    order = 0
    for term in description_terms:
        value = str(term or "").strip()
        key = value.lower()
        if not key:
            continue
        if key not in rank:
            rank[key] = (2, order, value)
        else:
            rank[key] = (rank[key][0] + 2, rank[key][1], rank[key][2])
        order += 1
    for term in document_terms:
        value = str(term or "").strip()
        key = value.lower()
        if not key:
            continue
        if key not in rank:
            rank[key] = (1, order, value)
        else:
            rank[key] = (rank[key][0] + 1, rank[key][1], rank[key][2])
        order += 1
    ordered = sorted(rank.values(), key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ordered]


def _enrich_text_filter(base_text: str, doc_text: str) -> str:
    """Append missing high-signal doc terms without overriding base filter intent."""
    base = str(base_text or "").strip()
    doc = str(doc_text or "").strip()
    if not base:
        return doc
    if not doc:
        return base
    base_tokens = set(_tokenize(base))
    doc_tokens = [t for t in _tokenize(doc) if t not in base_tokens]
    if not doc_tokens:
        return base
    return f"{base} {' '.join(doc_tokens[:6])}".strip()


def merge_intents(description_intent: dict, document_intent: dict) -> dict:
    """Merge description-base intent with additive document intent enrichment."""
    base = _as_mapping(description_intent)
    doc = _as_mapping(document_intent)

    base_filters = _as_mapping(base.get("search_filters"))
    doc_filters = _as_mapping(doc.get("search_filters"))
    merged_filters = dict(base_filters)

    for key in ("news", "papers", "patents"):
        merged_filters[key] = _enrich_text_filter(
            base_filters.get(key, ""), doc_filters.get(key, "")
        )

    for key in ("tiktok", "social", "instagram"):
        merged_filters[key] = str(base_filters.get(key) or doc_filters.get(key) or "").strip()

    merged_must_match = _dedupe_keep_order(
        [
            *_as_str_list(base.get("must_match_terms")),
            *_as_str_list(doc.get("must_match_terms")),
        ]
    )

    merged_entities = _weighted_union(
        _as_str_list(base.get("entities")),
        _as_str_list(doc.get("entities")),
    )
    merged_keywords = _weighted_union(
        _as_str_list(base.get("keywords")),
        _as_str_list(doc.get("keywords")),
    )
    merged_domain_terms = _weighted_union(
        _as_str_list(base.get("domain_terms")),
        _as_str_list(doc.get("domain_terms")),
    )

    merged = {
        **doc,
        **base,
        "domain": str(base.get("domain") or doc.get("domain") or "general"),
        "must_match_terms": merged_must_match,
        "entities": merged_entities,
        "keywords": merged_keywords[:12],
        "domain_terms": merged_domain_terms,
        "search_filters": merged_filters,
    }

    merged["search_filters"]["social"] = str(
        merged["search_filters"].get("tiktok") or merged["search_filters"].get("social") or ""
    )

    return _postprocess_intent(merged, description=str(base.get("description") or ""))


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _dedupe_keep_order(values: list[str]) -> list[str]:
    """Return case-insensitive deduplicated values preserving first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = value.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(value.strip())
    return out


def _as_mapping(value: object) -> dict:
    """Coerce possibly malformed LLM field values into a mapping."""
    return value if isinstance(value, dict) else {}


def _as_str_list(value: object) -> list[str]:
    """Coerce scalar/list intent fields into a list of non-empty strings."""
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _expand_related_keywords(domain: str, seed_keywords: list[str]) -> list[str]:
    """
    Expand domain keywords with curated related terms to avoid generic outputs.

    Expansion is deterministic and bounded, so behavior is stable across runs.
    """
    related: list[str] = []
    for domain_key, hints in _RELATED_KEYWORD_HINTS.items():
        if domain_key in domain:
            related.extend(hints)
    return related


def _keyword_quality_filter(keywords: list[str]) -> list[str]:
    """Remove generic or too-short keyword terms."""
    filtered: list[str] = []
    for kw in keywords:
        item = kw.strip()
        if not item:
            continue
        lower = item.lower()
        if lower in _GENERIC_KEYWORDS or lower in _STOPWORD_KEYWORDS:
            continue
        if len(_tokenize(item)) == 0:
            continue
        filtered.append(item)
    return filtered


def _normalize_phrase_key(value: str) -> str:
    """Return a normalized key for case-insensitive phrase deduplication."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip()


def _clean_phrase_list(value: object, *, max_items: int = 12) -> list[str]:
    """Preserve useful user/LLM phrases while removing empty or duplicate items."""
    if not isinstance(value, list | tuple | set):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").replace("#", " ").strip()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        key = _normalize_phrase_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _normalize_domain(domain: str, keywords: list[str], description: str) -> str:
    """Normalize domain label to snake_case and infer from context when missing."""
    raw = str(domain or "").strip().lower().replace(" ", "_")
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if raw:
        return raw

    candidates = _tokenize(" ".join(keywords))
    if not candidates:
        candidates = _tokenize(description)
    return "_".join(candidates[:3]) if candidates else "general"


def _derive_hashtag_terms(keywords: list[str], description: str) -> list[str]:
    """Derive compact, high-signal hashtag terms from keywords/description."""
    blocked = _STOPWORD_KEYWORDS | _GENERIC_KEYWORDS
    out: list[str] = []
    seen: set[str] = set()

    for phrase in keywords:
        raw_parts = _tokenize(phrase)
        if len(raw_parts) > 1:
            parts = [token for token in raw_parts if token not in blocked and len(token) >= 1]
        else:
            parts = [token for token in raw_parts if token not in blocked and len(token) >= 2]
        if not parts:
            continue
        compact = "".join(parts) if len(parts) > 1 else parts[0]
        if compact in seen or len(compact) < 3:
            continue
        seen.add(compact)
        out.append(compact)
        if len(out) >= 4:
            return out

    if len(out) >= 3:
        return out

    source_tokens = _tokenize(description)
    for token in source_tokens:
        if token in blocked or len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 4:
            break
    return out


def _order_keywords_for_intent(keywords: list[str]) -> list[str]:
    """Prioritize anchor and product terms ahead of generic analysis modifiers."""
    ranked: list[tuple[int, int, str]] = []
    for index, keyword in enumerate(keywords):
        text = str(keyword or "").strip()
        if not text:
            continue
        lowered = text.lower()
        tokens = _tokenize(lowered)
        has_compound_shape = " " in lowered or any(len(token) <= 2 for token in tokens)
        is_analysis_modifier = lowered in _ANALYSIS_MODIFIER_KEYWORDS
        if has_compound_shape and is_analysis_modifier:
            priority = 2
        elif has_compound_shape:
            priority = 0
        elif is_analysis_modifier:
            priority = 2
        else:
            priority = 1
        ranked.append((priority, index, text))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


def _enforce_dynamic_filters(search_filters: dict, keywords: list[str], description: str) -> dict:
    """Normalize filters without forcing any fixed domain vocabulary."""
    filters = dict(search_filters)
    keyword_phrase = " ".join(keywords[:6]).strip()
    description_phrase = " ".join(_tokenize(description)[:8]).strip()
    fallback_phrase = keyword_phrase or description_phrase

    for field in ("news", "papers", "patents"):
        base = str(filters.get(field) or "").strip()
        if not base:
            base = fallback_phrase
        if field == "papers":
            base = base.replace("#", " ")
        filters[field] = re.sub(r"\s+", " ", base).strip()

    hashtags = _derive_hashtag_terms(keywords, description)
    tiktok = str(filters.get("tiktok") or "").strip()
    if not tiktok and hashtags:
        tiktok = " ".join(f"#{term}" for term in hashtags)
    filters["tiktok"] = tiktok
    filters["social"] = tiktok

    instagram = str(filters.get("instagram") or "").replace("#", " ").strip()
    if not instagram:
        instagram = " ".join(_tokenize(keyword_phrase)[:3]).strip()
    words = instagram.split()
    filters["instagram"] = " ".join(words[:3]).strip()
    return filters


def _build_minimum_keywords(
    *,
    description: str,
    domain: str,
    entities: list[str],
    domain_terms: list[str],
    llm_keywords: list[str],
    search_filters: dict,
) -> list[str]:
    """
    Build a robust 5-10 keyword list with related terms and deterministic fallback.
    """
    filter_tokens = [
        search_filters.get("news", ""),
        search_filters.get("papers", ""),
        search_filters.get("patents", ""),
        search_filters.get("instagram", ""),
    ]
    description_terms = extract_signal_terms(description, max_terms=20)
    filter_terms = extract_signal_terms(" ".join(str(x) for x in filter_tokens), max_terms=20)
    entity_terms = _keyword_quality_filter(entities)
    domain_seed_terms = _keyword_quality_filter(domain_terms)
    seeds = _keyword_quality_filter(llm_keywords)
    candidate_terms = seeds + entity_terms + domain_seed_terms
    candidate_terms = _dedupe_keep_order(candidate_terms)
    candidate_terms = _keyword_quality_filter(candidate_terms)
    if len(candidate_terms) < 5:
        expansion_seed_terms = candidate_terms or filter_terms or description_terms
        related = _expand_related_keywords(domain, expansion_seed_terms)
        candidate_terms = candidate_terms + related + filter_terms + description_terms
        candidate_terms = _dedupe_keep_order(candidate_terms)
        candidate_terms = _keyword_quality_filter(candidate_terms)
    return candidate_terms[:10]


def _build_must_match_terms(
    *,
    explicit_terms: list[str],
    entities: list[str],
    keywords: list[str],
    domain_terms: list[str],
    description: str,
) -> list[str]:
    """Preserve explicit anchors when present; derive only when they are missing."""
    if explicit_terms:
        merged_explicit = _clean_phrase_list([*entities, *explicit_terms], max_items=6)
        if len(merged_explicit) >= 2:
            return merged_explicit[:6]

        derived_tail = derive_must_match_terms(
            explicit_terms=[],
            entities=entities,
            keywords=keywords,
            domain_terms=domain_terms,
            description=description,
            min_terms=3,
            max_terms=6,
        )
        merged = _clean_phrase_list([*merged_explicit, *derived_tail], max_items=6)
        return merged

    return derive_must_match_terms(
        explicit_terms=[],
        entities=entities,
        keywords=keywords,
        domain_terms=domain_terms,
        description=description,
        min_terms=3,
        max_terms=6,
    )


def _postprocess_intent(intent: dict, description: str) -> dict:
    """
    Enforce intent invariants required by ingestion/search quality.

    Invariants:
    - keywords must be 5-10 meaningful terms (non-generic where possible)
    - social filter must mirror tiktok
    - instagram filter must remain short (max 3 words)
    - papers filter must be natural language (no hashtags)
    """
    search_filters = dict(intent.get("search_filters") or {})

    instagram_text = str(search_filters.get("instagram") or "").replace("#", " ").strip()
    instagram_words = instagram_text.split()
    if len(instagram_words) > 3:
        instagram_text = " ".join(instagram_words[:3])
    search_filters["instagram"] = instagram_text

    tiktok_filter = str(search_filters.get("tiktok") or "").strip()
    search_filters["social"] = tiktok_filter

    papers_text = str(search_filters.get("papers") or "").replace("#", " ").strip()
    search_filters["papers"] = papers_text

    entities = _clean_phrase_list(intent.get("entities"), max_items=8)
    domain_terms = _clean_phrase_list(intent.get("domain_terms"), max_items=12)
    explicit_keywords = _clean_phrase_list(intent.get("keywords"), max_items=12)
    explicit_must_match = _clean_phrase_list(intent.get("must_match_terms"), max_items=6)

    normalized_keywords = _build_minimum_keywords(
        description=description,
        domain=str(intent.get("domain") or "general"),
        entities=entities,
        domain_terms=domain_terms,
        llm_keywords=explicit_keywords,
        search_filters=search_filters,
    )
    if not explicit_keywords:
        normalized_keywords = _order_keywords_for_intent(normalized_keywords)
    normalized_domain = _normalize_domain(
        str(intent.get("domain") or "general"),
        normalized_keywords,
        description,
    )
    search_filters = _enforce_dynamic_filters(search_filters, normalized_keywords, description)
    must_match_terms = _build_must_match_terms(
        explicit_terms=explicit_must_match,
        entities=entities,
        keywords=normalized_keywords,
        domain_terms=domain_terms,
        description=description,
    )

    return {
        **intent,
        "domain": normalized_domain,
        "entities": entities,
        "domain_terms": domain_terms,
        "keywords": normalized_keywords,
        "must_match_terms": must_match_terms,
        "search_filters": search_filters,
    }
