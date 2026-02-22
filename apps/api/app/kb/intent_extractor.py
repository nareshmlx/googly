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
You are a domain-expert beauty and cosmetics research analyst.
Extract structured research intent from a project description.
Output ONLY valid JSON — no prose, no markdown fences.

Before producing JSON, reason through these questions internally:
1. What is the most specific BEAUTY domain label? (e.g. "cosmetic_formulation" not "beauty"; "fragrance_trends" not "general")
2. What brands, companies, or product lines are relevant? (expand beyond the user's words — e.g. "P&G" → Olay, SK-II, Pantene, Gillette)
3. What active compounds, ingredients, or technical terms are central? (e.g. "active ingredients" → retinol, niacinamide, hyaluronic acid, peptides, AHA, BHA)
4. What practitioners or creators discuss this topic? (dermatologists, cosmetic chemists, formulators, beauty editors)
5. What academic/journal terms would appear in a paper abstract about this topic?
6. What hashtags do practitioners actually use on TikTok and Instagram — not the user's words, but what experts in this field post under?
7. What 1–3 word descriptive phrase best characterises this space for an Instagram user search? (This is used to find relevant accounts — keep it short and natural, like "skincare science" or "beauty chemistry")

Schema:
{
  "domain": "<most specific beauty/cosmetics/fragrance snake_case label, e.g. cosmetic_formulation, makeup_trends, fragrance_market>",
  "keywords": ["<expanded keyword1>", "<expanded keyword2>", ...],
  "search_filters": {
    "news": "<natural language query for news APIs — expanded brand/product names, no hashtags>",
    "papers": "<natural language query for academic paper APIs — use technical/scientific terms, no hashtags, no # symbols>",
    "patents": "<natural language query for patent APIs — company names + technical terms, no hashtags>",
    "tiktok": "<hashtags for TikTok search — what practitioners actually use, e.g. #skincare #retinol #niacinamide #cosmeticchemist>",
    "social": "<same as tiktok — kept for backwards compatibility>",
    "instagram": "<1–3 word descriptive phrase for Instagram user search, e.g. 'skincare science' or 'beauty chemistry' — NO hashtags, NO long strings>"
  },
  "confidence": <float 0.0–1.0>
}

Rules:
- domain: most specific label possible, always snake_case — think like a librarian classifying a journal
- keywords: 5–10 terms — expand beyond the user's words to expert vocabulary (brands, compounds, techniques)
- Stay strictly in beauty/cosmetics/fragrance/haircare/personal-care scope.
- If the prompt is ambiguous, still map to the nearest beauty/cosmetics/fragrance domain.
- search_filters.papers: natural language only — no hashtags, no # symbols, no boolean operators
- search_filters.tiktok: hashtag format (#) — use what practitioners post under, not what the user typed
- search_filters.social: exact copy of tiktok (backwards compatibility)
- search_filters.instagram: SHORT — 1–3 words max, descriptive, no hashtags — used for EnsembleData user search
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
    "beauty",
    "fashion",
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
    Extract structured intent from a project description using gpt-5-mini.

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
            model="gpt-5-mini",
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
    except Exception:
        logger.exception("intent_extractor.extract.error")
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
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=30.0,
        )
        text = response.choices[0].message.content or ""
        refined = json.loads(text.strip())
        # refined_at is always Python-generated — never trust the LLM to produce timestamps.
        refined["refined_at"] = datetime.now(UTC).isoformat()
        logger.info(
            "intent_extractor.refine.success",
            new_domain=refined.get("domain"),
            keyword_count=len(refined.get("keywords", [])),
        )
        # Deep-merge search_filters: LLM may omit sub-keys it didn't change,
        # which would silently delete tiktok/social/instagram on a shallow merge.
        existing_filters = existing_intent.get("search_filters") or {}
        refined_filters = refined.get("search_filters") or {}
        merged_filters = {**existing_filters, **refined_filters}
        # social must always mirror tiktok — unconditional, never diverge.
        merged_filters["social"] = merged_filters.get("tiktok") or merged_filters.get("social", "")
        refined["search_filters"] = merged_filters
        return _postprocess_intent({**existing_intent, **refined}, description="")
    except json.JSONDecodeError:
        logger.warning("intent_extractor.refine.parse_error")
        return existing_intent
    except Exception:
        logger.exception("intent_extractor.refine.error")
        return existing_intent


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
        if any(token in seed_joined for token in ("retinol", "niacinamide", "skincare", "cosmetic")):
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


def _is_beauty_keyword(keyword: str, description_tokens: set[str]) -> bool:
    """Return True if keyword is within beauty/cosmetics/fragrance scope."""
    tokens = set(_tokenize(keyword))
    if not tokens:
        return False
    if tokens & _BEAUTY_SCOPE_TERMS:
        return True
    if tokens & description_tokens:
        return True
    compact = re.sub(r"[^a-z0-9]", "", keyword.lower())
    return compact in description_tokens


def _enforce_beauty_keywords(keywords: list[str], description: str) -> list[str]:
    """Keep beauty-scope keywords and backfill with beauty defaults."""
    description_tokens = set(_tokenize(description))
    scoped = [kw for kw in keywords if _is_beauty_keyword(kw, description_tokens)]
    scoped = _dedupe_keep_order(scoped)
    if len(scoped) < 5:
        scoped.extend(_BEAUTY_FALLBACK_KEYWORDS)
        scoped = _dedupe_keep_order(scoped)
    return scoped[:10]


def _normalize_beauty_domain(domain: str, keywords: list[str], description: str) -> str:
    """Map any out-of-scope domain label to a beauty-focused fallback domain."""
    value = (domain or "").strip().lower().replace(" ", "_")
    domain_tokens = set(_tokenize(value))
    if domain_tokens & _BEAUTY_SCOPE_TERMS:
        return value or "beauty_market_intelligence"
    return "beauty_market_intelligence"


def _enforce_beauty_filters(search_filters: dict, keywords: list[str]) -> dict:
    """Ensure search filters stay in beauty/cosmetics/fragrance scope."""
    filters = dict(search_filters)
    keyword_phrase = " ".join(keywords[:6]).strip()
    beauty_context = "beauty cosmetics skincare fragrance"

    for field in ("news", "papers", "patents"):
        base = str(filters.get(field) or "").strip()
        base_tokens = set(_tokenize(base))
        if not base or not (base_tokens & _BEAUTY_SCOPE_TERMS):
            base = keyword_phrase or beauty_context
        if beauty_context not in base.lower():
            base = f"{base} {beauty_context}".strip()
        filters[field] = base

    tiktok = str(filters.get("tiktok") or "").strip()
    if not tiktok:
        tiktok = "#beauty #skincare #makeup #fragrance"
    if not (set(_tokenize(tiktok)) & _BEAUTY_SCOPE_TERMS):
        tiktok = "#beauty #skincare #makeup #fragrance"
    elif "#beauty" not in tiktok.lower():
        tiktok = f"{tiktok} #beauty"
    filters["tiktok"] = tiktok
    filters["social"] = tiktok

    instagram = str(filters.get("instagram") or "").replace("#", " ").strip()
    words = instagram.split()
    if len(words) > 3:
        instagram = " ".join(words[:3])
    if not instagram or not (set(_tokenize(instagram)) & _BEAUTY_SCOPE_TERMS):
        instagram = "beauty trends"
    filters["instagram"] = instagram
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
    description_terms = [t for t in _tokenize(description) if t in _BEAUTY_SCOPE_TERMS][:20]
    filter_terms = [t for t in _tokenize(" ".join(str(x) for x in filter_tokens)) if t in _BEAUTY_SCOPE_TERMS][:20]
    candidate_terms = seeds + related + filter_terms + description_terms
    candidate_terms = _dedupe_keep_order(candidate_terms)
    candidate_terms = _keyword_quality_filter(candidate_terms)
    if len(candidate_terms) < 5:
        candidate_terms.extend(_RELATED_KEYWORD_HINTS.get("skincare", []))
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
    normalized_keywords = _enforce_beauty_keywords(normalized_keywords, description)
    normalized_domain = _normalize_beauty_domain(
        str(intent.get("domain") or "general"),
        normalized_keywords,
        description,
    )
    search_filters = _enforce_beauty_filters(search_filters, normalized_keywords)

    return {
        **intent,
        "domain": normalized_domain,
        "keywords": normalized_keywords,
        "search_filters": search_filters,
    }
