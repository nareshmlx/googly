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
from openai import AsyncOpenAI

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Module-level singleton — avoids re-creating the httpx connection pool on every call.
_openai_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Return (or lazily create) the module-level AsyncOpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


_EXTRACT_SYSTEM = """\
You are a domain-agnostic research intent analyst.
Extract structured research intent from a project description.
Output ONLY valid JSON — no prose, no markdown fences.

Before producing JSON, reason through these questions internally:
1. What is the most specific domain label? (e.g. "cosmetic_formulation", "battery_materials", "llm_evaluation")
2. What named entities are relevant? (brands, companies, products, compounds, frameworks, standards)
3. What technical terms are central?
4. What practitioners or creators discuss this topic?
5. What academic/journal terms would appear in a paper abstract about this topic?
6. What hashtags are actually used on social platforms for this topic?
7. What 1–3 word descriptive phrase best characterises this space for Instagram account search?

Schema:
{
  "domain": "<specific snake_case domain label, e.g. cosmetic_formulation, battery_recycling, llm_safety>",
  "keywords": ["<expanded keyword1>", "<expanded keyword2>", ...],
  "search_filters": {
    "news": "<natural language query for news APIs — domain terms/entities, no hashtags>",
    "papers": "<natural language query for academic paper APIs — technical/scientific terms, no hashtags, no # symbols>",
    "patents": "<natural language query for patent APIs — company names + technical terms, no hashtags>",
    "tiktok": "<hashtags for TikTok search when applicable; may be empty if topic is not social-native>",
    "social": "<same as tiktok — kept for backwards compatibility>",
    "instagram": "<1–3 word phrase for Instagram account search, no hashtags>"
  },
  "confidence": <float 0.0–1.0>
}

Rules:
- domain: most specific label possible, always snake_case — think like a librarian classifying a journal
- keywords: 5–10 terms — expand beyond the user's words to expert vocabulary
- Never force a domain that is not present in the user description or sample text.
- search_filters.papers: natural language only — no hashtags, no # symbols, no boolean operators
- search_filters.tiktok: hashtag format (#) when used; keep concise and topic-focused
- search_filters.social: exact copy of tiktok (backwards compatibility)
- search_filters.instagram: SHORT — 1–3 words max, descriptive, no hashtags
- confidence: 1.0 = unambiguous domain, 0.5 = could be multiple domains
- Output ONLY the JSON object, nothing else
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
- Merge keywords additively — union of existing + new, deduplicated, max 12 total
- Update search_filters only if new keywords materially change the meaning of the search
- Preserve all existing search_filters fields including tiktok, social, and instagram
- search_filters.social must always equal search_filters.tiktok (backwards compatibility)
- search_filters.instagram must remain SHORT — 1–3 words max, no hashtags
- search_filters.papers must remain natural language only — no hashtags, no # symbols
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

_BEAUTY_SCOPE_TERMS: set[str] = {
    "beauty",
    "cosmetic",
    "cosmetics",
    "skincare",
    "skin",
    "haircare",
    "hair",
    "makeup",
    "fragrance",
    "perfume",
    "deodorant",
    "moisturizer",
    "sunscreen",
    "balm",
    "serum",
    "toner",
    "exfoliant",
    "acne",
    "pigmentation",
    "bodycare",
    "cleanser",
    "foundation",
    "concealer",
    "lipstick",
    "lip",
    "lipbalm",
    "lipgloss",
    "mascara",
    "eyeliner",
    "blush",
    "bronzer",
    "primer",
    "retinol",
    "niacinamide",
    "hyaluronic",
    "ceramide",
    "peptide",
    "aha",
    "bha",
    "spf",
    "parfum",
}

_BEAUTY_FALLBACK_KEYWORDS: list[str] = [
    "skincare",
    "makeup",
    "fragrance",
    "retinol",
    "niacinamide",
    "hyaluronic acid",
    "peptides",
]

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
    default_filters: dict[str, str] = {
        "news": description[:80],
        "papers": description[:80],
        "patents": description[:80],
        "tiktok": "",
        "social": "",
        "instagram": "",
    }
    default: dict[str, Any] = {
        "domain": "general",
        "keywords": [],
        "search_filters": default_filters,
        "confidence": 0.0,
    }

    if not settings.OPENAI_API_KEY:
        logger.warning("intent_extractor.no_api_key")
        return _postprocess_intent(default, description=description)

    prompt = f"Project description: {description}"
    if sample_text:
        prompt += f"\n\nSample document text (first 500 chars):\n{sample_text[:500]}"

    logger.info("intent_extractor.extract.start", description_preview=description[:60])

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.INTENT_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=30.0,
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
        result = _postprocess_intent(result, description=description)
        return result
    except json.JSONDecodeError:
        logger.warning(
            "intent_extractor.extract.parse_error",
            description_preview=description[:60],
        )
        return _postprocess_intent(default, description=description)
    except Exception as exc:
        logger.error(
            "intent_extractor.extract.error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return _postprocess_intent(default, description=description)


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
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.INTENT_MODEL,
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=30.0,
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
    if not related:
        seed_joined = " ".join(seed_keywords).lower()
        if any(
            token in seed_joined for token in ("retinol", "niacinamide", "skincare", "cosmetic")
        ):
            related.extend(_RELATED_KEYWORD_HINTS["skincare"])
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
    source_tokens = _tokenize(" ".join(keywords))
    if not source_tokens:
        source_tokens = _tokenize(description)
    out: list[str] = []
    seen: set[str] = set()
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
    llm_keywords: list[str],
    search_filters: dict,
) -> list[str]:
    """
    Build a robust 5–10 keyword list with related terms and deterministic fallback.
    """
    seeds = _keyword_quality_filter(llm_keywords)
    related = _expand_related_keywords(domain, seeds)
    filter_tokens = [
        search_filters.get("news", ""),
        search_filters.get("papers", ""),
        search_filters.get("patents", ""),
        search_filters.get("instagram", ""),
    ]
    description_terms = _tokenize(description)[:20]
    filter_terms = _tokenize(" ".join(str(x) for x in filter_tokens))[:20]
    candidate_terms = seeds + related + filter_terms + description_terms
    candidate_terms = _dedupe_keep_order(candidate_terms)
    candidate_terms = _keyword_quality_filter(candidate_terms)
    if len(candidate_terms) < 5:
        candidate_terms.extend(_tokenize(description)[:8])
        candidate_terms = _dedupe_keep_order(candidate_terms)
        candidate_terms = _keyword_quality_filter(candidate_terms)
    return candidate_terms[:10]


def _postprocess_intent(intent: dict, description: str) -> dict:
    """
    Enforce intent invariants required by ingestion/search quality.

    Invariants:
    - keywords must be 5–10 meaningful terms (non-generic where possible)
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

    keywords = intent.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords]
    normalized_keywords = _build_minimum_keywords(
        description=description,
        domain=str(intent.get("domain") or "general"),
        llm_keywords=keywords,
        search_filters=search_filters,
    )
    normalized_domain = _normalize_domain(
        str(intent.get("domain") or "general"),
        normalized_keywords,
        description,
    )
    search_filters = _enforce_dynamic_filters(search_filters, normalized_keywords, description)

    return {
        **intent,
        "domain": normalized_domain,
        "keywords": normalized_keywords,
        "search_filters": search_filters,
    }
