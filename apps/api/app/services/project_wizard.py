"""Project wizard service helpers for dynamic Q&A and schema-safe intent merging."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from statistics import mean
from typing import Any

import structlog

from app.core.config import settings
from app.core.openai_client import get_openai_client, openai_completions_with_circuit_breaker
from app.core.query_sanitize import (
    derive_must_match_terms,
)

logger = structlog.get_logger(__name__)

_CORE_DIMENSIONS: tuple[str, ...] = (
    "objective_clarity",
    "pain_point_clarity",
    "output_clarity",
)
_DOMAIN_DIMENSION = "domain_specificity"
_ALL_DIMENSIONS: tuple[str, ...] = (*_CORE_DIMENSIONS, _DOMAIN_DIMENSION)
_CORE_STOP_THRESHOLD = 0.7
_DOMAIN_STOP_THRESHOLD = 0.6

_DEFAULT_TIME_HORIZON = "last 1 year"

_DEFAULT_SOURCE_TOGGLES: dict[str, bool] = {
    "tiktok": True,
    "instagram": True,
    "youtube": True,
    "reddit": True,
    "x": True,
    "papers": True,
    "patents": True,
    "news": True,
    "web_tavily": True,
    "web_exa": True,
}

_QUESTION_CATALOG: dict[str, tuple[str, ...]] = {
    "objective_clarity": (
        "What exact research question should this project answer, and what decision will it support?",
        "What is the single most important decision this research should enable in the next planning cycle?",
        "If this project succeeds, what precise question will no longer be ambiguous for your team?",
    ),
    "pain_point_clarity": (
        "What specific pain point, risk, or uncertainty are you trying to reduce with this research?",
        "What is the current consequence of not having this insight (for example delays, wasted spend, or poor prioritization)?",
        "Where is your team currently guessing instead of using evidence, and what risk does that create?",
    ),
    "output_clarity": (
        "What should the final output look like (for example: ranked opportunities, competitor matrix, or action plan)?",
        "What deliverable format would be most useful to decision makers (for example one-page brief, dashboard, or prioritized recommendations)?",
        "How should findings be structured so stakeholders can act immediately?",
    ),
    "domain_specificity": (
        "Which data platforms should be prioritized first (for example TikTok, Reddit, OpenAlex, Patents, News, Web)?",
        "Which channels or evidence sources matter most for this decision, and which can be deprioritized?",
        "Are there specific communities, publishers, datasets, or regions we should explicitly include or exclude?",
    ),
}

_EVALUATE_SYSTEM_PROMPT = """\
You evaluate research-project sufficiency for an intake wizard.
Return JSON only.

Required schema:
{
  "scores": {
    "objective_clarity": <0.0-1.0>,
    "pain_point_clarity": <0.0-1.0>,
    "output_clarity": <0.0-1.0>,
    "domain_specificity": <0.0-1.0>
  },
  "weakest_dimension": "objective_clarity|pain_point_clarity|output_clarity|domain_specificity",
  "next_question": "<single focused follow-up question>"
}

Rules:
- Ask one clear question only.
- Avoid repeating previously asked question wording; if revisiting a dimension, ask from a different angle.
- Scores should reflect current project context with conservative confidence.
- Domain specificity should stay low if source/platform preferences are not explicit.
"""

_SYNTHESIZE_SYSTEM_PROMPT = """\
You synthesize wizard context into a review payload for project creation.
Return JSON only.

Schema:
{
  "enriched_description": "<plain-text multi-section brief with explicit headings>",
  "domain_focus": "<short domain phrase>",
  "key_entities": ["..."],
  "must_match_terms": ["..."],
  "time_horizon": "<short phrase>",
  "target_sources": {
    "tiktok": true,
    "instagram": true,
    "youtube": true,
    "reddit": true,
    "x": true,
    "papers": true,
    "patents": true,
    "news": true,
    "web_tavily": true,
    "web_exa": true
  }
}

Rules:
- Preserve explicit user constraints from Q&A.
- Keep lists concise and specific.
- Use these section headings in this exact order inside enriched_description:
  Project Goal (Research Goal):
  User Pain Points:
  Primary Users / Stakeholders:
  Key Decisions and Output Format:
  Scope and Evidence Priorities:
  Constraints and Assumptions:
  Success Signals and Next Steps:
- Keep enriched_description plain text (no markdown syntax).
"""

_MERGE_SYSTEM_PROMPT = """\
You improve values in an existing structured_intent while preserving exact schema.
Return JSON only.

Rules:
- Keep the SAME keys and nesting as structured_intent.
- Do NOT add new keys.
- Do NOT remove keys.
- Update values to align with enriched_description + overrides.
- If a key cannot be improved, keep its current value.
"""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _snake_case(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "general"


def _clamp_score(value: object) -> float:
    if not isinstance(value, int | float | str):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _as_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_list(value: object, *, max_items: int = 12) -> list[str]:
    if not isinstance(value, list | tuple | set):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def build_intent_seed_text(
    *,
    description: str,
    domain_focus: str = "",
    key_entities: object = None,
    must_match_terms: object = None,
) -> str:
    """Build a concise extraction seed from the core query plus explicit anchors."""
    base_description = str(description or "").strip()
    seed_parts: list[str] = []

    if base_description:
        seed_parts.append(base_description)

    for value in [str(domain_focus or "").strip(), *_clean_list(key_entities), *_clean_list(must_match_terms)]:
        if not value:
            continue
        lowered_value = value.lower()
        if any(lowered_value in part.lower() for part in seed_parts):
            continue
        seed_parts.append(value)

    return " ".join(seed_parts).strip() or base_description


def _normalize_qa_pairs(qa_pairs: list[dict] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in qa_pairs or []:
        item = _as_mapping(row)
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        dimension = str(item.get("dimension") or "").strip()
        if not question or not answer:
            continue
        out.append({"question": question, "answer": answer, "dimension": dimension})
    return out


def _normalize_source_toggles(source_toggles: Mapping[str, object] | None) -> dict[str, bool]:
    normalized = dict(_DEFAULT_SOURCE_TOGGLES)
    if not isinstance(source_toggles, Mapping):
        return normalized

    incoming = {str(k): bool(v) for k, v in source_toggles.items()}
    if "web" in incoming:
        normalized["web_tavily"] = bool(incoming["web"])
        normalized["web_exa"] = bool(incoming["web"])

    for key in normalized:
        if key in incoming:
            normalized[key] = bool(incoming[key])
    return normalized


def _normalize_question_text(value: str) -> str:
    return " ".join(_tokenize(value))


def _fallback_question(dimension: str, qa_pairs: list[dict[str, str]] | None = None) -> str:
    templates = _QUESTION_CATALOG.get(dimension) or _QUESTION_CATALOG["objective_clarity"]
    normalized_pairs = qa_pairs or []
    asked_fingerprints = {
        _normalize_question_text(str(row.get("question") or ""))
        for row in normalized_pairs
        if str(row.get("question") or "").strip()
    }

    for template in templates:
        if _normalize_question_text(template) not in asked_fingerprints:
            return template

    dimension_count = sum(
        1 for row in normalized_pairs if str(row.get("dimension") or "").strip() == dimension
    )
    return templates[dimension_count % len(templates)]


def _asked_dimension_counts(qa_pairs: list[dict[str, str]]) -> dict[str, int]:
    counts = {key: 0 for key in _ALL_DIMENSIONS}
    for row in qa_pairs:
        dimension = str(row.get("dimension") or "").strip()
        if dimension in counts:
            counts[dimension] += 1
    return counts


def _pick_next_dimension(
    *,
    candidates: list[str],
    scores: Mapping[str, float],
    qa_pairs: list[dict[str, str]],
) -> str | None:
    if not candidates:
        return None

    counts = _asked_dimension_counts(qa_pairs)
    last_dimension = ""
    for row in reversed(qa_pairs):
        dimension = str(row.get("dimension") or "").strip()
        if dimension in _ALL_DIMENSIONS:
            last_dimension = dimension
            break

    ordered = sorted(
        candidates,
        key=lambda key: (counts.get(key, 0), scores.get(key, 0.0)),
    )
    if last_dimension and len(ordered) > 1:
        alternatives = [key for key in ordered if key != last_dimension]
        if alternatives:
            return alternatives[0]
    return ordered[0]


def _heuristic_scores(
    *,
    title: str,
    description: str,
    qa_pairs: list[dict[str, str]],
) -> dict[str, float]:
    text = " ".join([title, description, *[p["answer"] for p in qa_pairs]]).strip()
    tokens = _tokenize(text)
    desc_tokens = _tokenize(description)
    qa_count = len(qa_pairs)

    pain_words = {
        "problem",
        "pain",
        "risk",
        "challenge",
        "issue",
        "uncertain",
        "unknown",
        "decline",
        "loss",
        "bottleneck",
    }
    output_words = {
        "report",
        "summary",
        "recommendation",
        "roadmap",
        "framework",
        "matrix",
        "dashboard",
        "plan",
        "comparison",
        "prioritize",
        "rank",
    }
    domain_words = {
        "tiktok",
        "reddit",
        "openalex",
        "patent",
        "patents",
        "news",
        "web",
        "instagram",
        "youtube",
        "x",
    }

    def _dimension_signal(dimension: str) -> float:
        answers = [
            row["answer"]
            for row in qa_pairs
            if str(row.get("dimension") or "").strip() == dimension
        ]
        if not answers:
            return 0.0
        answer_token_lengths = [len(_tokenize(answer)) for answer in answers]
        avg_length = mean(answer_token_lengths) if answer_token_lengths else 0.0
        richness = min(1.0, avg_length / 24.0)
        coverage = min(0.35, len(answers) * 0.12)
        return _clamp_score(richness + coverage)

    density = min(1.0, len(desc_tokens) / 40.0)
    qa_boost = min(0.3, qa_count * 0.1)
    objective_signal = _dimension_signal("objective_clarity")
    pain_signal = _dimension_signal("pain_point_clarity")
    output_signal = _dimension_signal("output_clarity")
    domain_signal = _dimension_signal("domain_specificity")

    objective = 0.32 + density * 0.42 + qa_boost + objective_signal * 0.22
    if any(term in tokens for term in ("goal", "objective", "decide", "decision", "evaluate")):
        objective += 0.08

    pain = 0.24 + density * 0.34 + qa_boost + pain_signal * 0.25
    if any(word in tokens for word in pain_words):
        pain += 0.12

    output = 0.24 + density * 0.34 + qa_boost + output_signal * 0.25
    if any(word in tokens for word in output_words):
        output += 0.12

    domain = 0.12 + qa_boost * 0.7 + domain_signal * 0.35
    if any(word in tokens for word in domain_words):
        domain += 0.2

    return {
        "objective_clarity": _clamp_score(objective),
        "pain_point_clarity": _clamp_score(pain),
        "output_clarity": _clamp_score(output),
        "domain_specificity": _clamp_score(domain),
    }


def _resolve_next_dimension(
    scores: Mapping[str, float],
    qa_pairs: list[dict[str, str]],
    max_questions: int,
) -> tuple[bool, str | None]:
    asked_count = len(qa_pairs)
    if asked_count >= max_questions:
        return True, None

    unresolved_core = [key for key in _CORE_DIMENSIONS if scores.get(key, 0.0) < _CORE_STOP_THRESHOLD]
    if unresolved_core:
        next_dimension = _pick_next_dimension(
            candidates=unresolved_core,
            scores=scores,
            qa_pairs=qa_pairs,
        )
        return False, next_dimension

    if scores.get(_DOMAIN_DIMENSION, 0.0) < _DOMAIN_STOP_THRESHOLD:
        return False, _DOMAIN_DIMENSION

    return True, None


async def _llm_json(
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any] | None:
    if not settings.OPENAI_API_KEY:
        return None

    try:
        client = get_openai_client()
        create_comp = openai_completions_with_circuit_breaker(client)
        response = await create_comp(
            model=settings.INTENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            timeout=settings.INTENT_EXTRACT_TIMEOUT_SECONDS,
        )
        payload = (response.choices[0].message.content or "").strip()
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception as exc:
        logger.warning("project_wizard.llm_json_failed", error=str(exc))
        return None


def _qa_transcript(title: str, description: str, qa_pairs: list[dict[str, str]]) -> str:
    qa_lines = "\n".join(
        [f"- Q: {item['question']}\n  A: {item['answer']}" for item in qa_pairs]
    )
    if not qa_lines:
        qa_lines = "- (none)"
    return (
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Answered Q&A:\n{qa_lines}"
    )


async def evaluate_project_sufficiency(
    *,
    title: str,
    description: str,
    qa_pairs: list[dict] | None = None,
    max_questions: int = 5,
) -> dict[str, Any]:
    """Score sufficiency dimensions and generate the next focused question."""
    normalized_pairs = _normalize_qa_pairs(qa_pairs)
    max_questions = max(1, min(10, int(max_questions or 5)))

    scores = _heuristic_scores(
        title=title,
        description=description,
        qa_pairs=normalized_pairs,
    )

    llm_payload = await _llm_json(
        system_prompt=_EVALUATE_SYSTEM_PROMPT,
        user_prompt=_qa_transcript(title, description, normalized_pairs),
    )
    if llm_payload:
        llm_scores = _as_mapping(llm_payload.get("scores"))
        for key in _ALL_DIMENSIONS:
            if key in llm_scores:
                llm_score = _clamp_score(llm_scores.get(key))
                scores[key] = max(scores.get(key, 0.0), llm_score)

    should_stop, next_dimension = _resolve_next_dimension(scores, normalized_pairs, max_questions)
    weakest_dimension = min(_ALL_DIMENSIONS, key=lambda key: scores.get(key, 0.0))

    next_question = None
    if not should_stop and next_dimension:
        next_question = _fallback_question(next_dimension, normalized_pairs)
        if llm_payload:
            llm_dimension = str(llm_payload.get("weakest_dimension") or "").strip()
            llm_question = str(llm_payload.get("next_question") or "").strip()
            asked_question_keys = {
                _normalize_question_text(pair["question"])
                for pair in normalized_pairs
                if pair["question"]
            }
            if (
                llm_dimension == next_dimension
                and llm_question
                and _normalize_question_text(llm_question) not in asked_question_keys
            ):
                next_question = llm_question

    overall = mean([scores.get(key, 0.0) for key in _CORE_DIMENSIONS])

    return {
        "scores": {key: _clamp_score(scores.get(key, 0.0)) for key in _ALL_DIMENSIONS},
        "overall_score": _clamp_score(overall),
        "weakest_dimension": weakest_dimension,
        "next_dimension": next_dimension,
        "next_question": next_question,
        "should_stop": bool(should_stop),
        "asked_questions": len(normalized_pairs),
        "max_questions": max_questions,
    }


def _fallback_synthesis(
    *,
    title: str,
    description: str,
    qa_pairs: list[dict[str, str]],
    structured_intent: Mapping[str, Any],
    source_toggles: dict[str, bool],
) -> dict[str, Any]:
    answers = " ".join(pair["answer"] for pair in qa_pairs).strip()
    intent_domain = str(structured_intent.get("domain") or "").replace("_", " ").strip()

    keywords = _clean_list(structured_intent.get("keywords"), max_items=8)
    entities = _clean_list(structured_intent.get("entities"), max_items=8)
    must_terms = _clean_list(structured_intent.get("must_match_terms"), max_items=8)
    domain_terms = _clean_list(structured_intent.get("domain_terms"), max_items=8)

    if not entities:
        entities = keywords[:4]
    if not must_terms:
        must_terms = derive_must_match_terms(
            explicit_terms=[],
            entities=entities,
            keywords=keywords,
            domain_terms=domain_terms,
            description=description,
            min_terms=3,
            max_terms=6,
        )

    domain_focus = intent_domain or "research focus"
    objective_answers = " ".join(
        pair["answer"]
        for pair in qa_pairs
        if str(pair.get("dimension") or "").strip() == "objective_clarity"
    ).strip()
    pain_answers = " ".join(
        pair["answer"]
        for pair in qa_pairs
        if str(pair.get("dimension") or "").strip() == "pain_point_clarity"
    ).strip()
    output_answers = " ".join(
        pair["answer"]
        for pair in qa_pairs
        if str(pair.get("dimension") or "").strip() == "output_clarity"
    ).strip()
    domain_answers = " ".join(
        pair["answer"]
        for pair in qa_pairs
        if str(pair.get("dimension") or "").strip() == "domain_specificity"
    ).strip()

    enabled_sources = [key for key, enabled in source_toggles.items() if enabled]
    source_summary = ", ".join(enabled_sources[:8]) if enabled_sources else "none selected"
    entity_summary = ", ".join(entities[:5]) if entities else "No explicit entities yet."
    must_term_summary = ", ".join(must_terms[:6]) if must_terms else "No strict must-match terms yet."

    project_goal = objective_answers or description.strip() or title.strip()
    pain_points = pain_answers or (
        "User pain points are partially specified; key uncertainty should be clarified in final review."
    )
    stakeholders = (
        f"Primary stakeholders likely include teams focused on: {entity_summary}"
        if entities
        else "Primary stakeholders: brand, insights, and strategy decision makers."
    )
    key_decisions = output_answers or (
        "Decision-ready output should include prioritized findings, trade-offs, and recommendations."
    )
    evidence_scope = domain_answers or (
        f"Prioritize sources: {source_summary}. Domain focus: {domain_focus}."
    )
    constraints = (
        f"Time horizon: {_DEFAULT_TIME_HORIZON}. Must-match terms: {must_term_summary}. "
        f"Additional wizard context: {answers or 'none provided'}"
    )
    next_steps = (
        "Success means stakeholders can act on a ranked set of opportunities and risks with clear evidence links."
    )

    enriched_description = "\n\n".join(
        [
            f"Project Goal (Research Goal): {project_goal}",
            f"User Pain Points: {pain_points}",
            f"Primary Users / Stakeholders: {stakeholders}",
            f"Key Decisions and Output Format: {key_decisions}",
            f"Scope and Evidence Priorities: {evidence_scope}",
            f"Constraints and Assumptions: {constraints}",
            f"Success Signals and Next Steps: {next_steps}",
        ]
    ).strip()

    return {
        "enriched_description": enriched_description,
        "domain_focus": domain_focus,
        "key_entities": entities,
        "must_match_terms": must_terms,
        "time_horizon": _DEFAULT_TIME_HORIZON,
        "target_sources": source_toggles,
    }


async def synthesize_wizard_review(
    *,
    title: str,
    description: str,
    qa_pairs: list[dict] | None = None,
    structured_intent: Mapping[str, Any] | None = None,
    source_toggles: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Create Phase-2 review payload with enriched description and editable defaults."""
    normalized_pairs = _normalize_qa_pairs(qa_pairs)
    safe_intent = _as_mapping(structured_intent)
    safe_sources = _normalize_source_toggles(source_toggles)

    fallback = _fallback_synthesis(
        title=title,
        description=description,
        qa_pairs=normalized_pairs,
        structured_intent=safe_intent,
        source_toggles=safe_sources,
    )

    llm_payload = await _llm_json(
        system_prompt=_SYNTHESIZE_SYSTEM_PROMPT,
        user_prompt=(
            f"{_qa_transcript(title, description, normalized_pairs)}\n\n"
            f"Base structured_intent: {json.dumps(safe_intent, ensure_ascii=True)}\n"
            f"Current source toggles: {json.dumps(safe_sources, ensure_ascii=True)}"
        ),
    )
    if not llm_payload:
        return fallback

    target_sources = _normalize_source_toggles(_as_mapping(llm_payload.get("target_sources")) or safe_sources)

    enriched_description = str(llm_payload.get("enriched_description") or "").strip()
    if not enriched_description:
        enriched_description = fallback["enriched_description"]

    domain_focus = str(llm_payload.get("domain_focus") or "").strip() or fallback["domain_focus"]
    key_entities = _clean_list(llm_payload.get("key_entities"), max_items=10) or fallback["key_entities"]
    must_match_terms = _clean_list(llm_payload.get("must_match_terms"), max_items=10) or fallback[
        "must_match_terms"
    ]
    time_horizon = str(llm_payload.get("time_horizon") or "").strip() or fallback["time_horizon"]

    return {
        "enriched_description": enriched_description,
        "domain_focus": domain_focus,
        "key_entities": key_entities,
        "must_match_terms": must_match_terms,
        "time_horizon": time_horizon,
        "target_sources": target_sources,
    }


def _merge_preserving_schema(base: Any, candidate: Any) -> Any:
    if isinstance(base, Mapping):
        candidate_map = candidate if isinstance(candidate, Mapping) else {}
        merged: dict[str, Any] = {}
        for key, base_value in base.items():
            merged[key] = _merge_preserving_schema(base_value, candidate_map.get(key, base_value))
        return merged

    if isinstance(base, list):
        if isinstance(candidate, list):
            if not base:
                return list(candidate)
            if all(isinstance(item, str) for item in base):
                return _clean_list(candidate, max_items=max(len(base), 12))
            return list(candidate)
        return list(base)

    if isinstance(base, bool):
        return bool(candidate)

    if isinstance(base, int) and not isinstance(base, bool):
        try:
            return int(candidate)
        except (TypeError, ValueError):
            return base

    if isinstance(base, float):
        return _clamp_score(candidate)

    if isinstance(base, str):
        text = str(candidate or "").strip()
        return text or base

    return candidate if candidate is not None else base


def _merge_text_terms(base_text: str, terms: list[str], time_horizon: str) -> str:
    text = str(base_text or "").strip()
    lower = text.lower()
    for term in terms:
        normalized = term.strip()
        if not normalized:
            continue
        if normalized.lower() in lower:
            continue
        text = f"{text} {normalized}".strip() if text else normalized
        lower = text.lower()
    if time_horizon and time_horizon.lower() not in lower:
        text = f"{text} {time_horizon}".strip() if text else time_horizon
    return text


def _apply_explicit_overrides(
    *,
    merged_intent: dict[str, Any],
    overrides: Mapping[str, Any],
    enriched_description: str,
) -> dict[str, Any]:
    domain_focus = str(overrides.get("domain_focus") or "").strip()
    key_entities = _clean_list(overrides.get("key_entities"), max_items=10)
    must_match_terms = _clean_list(overrides.get("must_match_terms"), max_items=10)
    time_horizon = str(overrides.get("time_horizon") or "").strip()
    target_sources = _normalize_source_toggles(_as_mapping(overrides.get("target_sources")))

    if domain_focus and "domain" in merged_intent:
        merged_intent["domain"] = _snake_case(domain_focus)

    if "keywords" in merged_intent and isinstance(merged_intent.get("keywords"), list):
        seeds = _clean_list(merged_intent.get("keywords"), max_items=16)
        merged_intent["keywords"] = _clean_list(
            [*key_entities, *must_match_terms, *seeds],
            max_items=16,
        )

    if "entities" in merged_intent and isinstance(merged_intent.get("entities"), list):
        merged_intent["entities"] = _clean_list(
            [*merged_intent.get("entities", []), *key_entities],
            max_items=12,
        )

    if "must_match_terms" in merged_intent and isinstance(merged_intent.get("must_match_terms"), list):
        explicit_must_match = _clean_list(
            [*must_match_terms, *merged_intent.get("must_match_terms", [])],
            max_items=12,
        )
        if must_match_terms:
            merged_intent["must_match_terms"] = _clean_list(
                [*must_match_terms, *key_entities],
                max_items=6,
            )
        else:
            merged_intent["must_match_terms"] = derive_must_match_terms(
                explicit_terms=explicit_must_match,
                entities=merged_intent.get("entities", []),
                keywords=merged_intent.get("keywords", []),
                domain_terms=merged_intent.get("domain_terms", []),
                description="",
                min_terms=3,
                max_terms=6,
            )

    filters = _as_mapping(merged_intent.get("search_filters"))
    if filters:
        prioritized_terms = _clean_list([*must_match_terms, *key_entities], max_items=6)
        for key in ("news", "papers", "patents"):
            if key in filters:
                merged_filter = _merge_text_terms(
                    str(filters.get(key) or ""),
                    prioritized_terms,
                    time_horizon,
                )
                filters[key] = re.sub(r"\s+", " ", merged_filter).strip()

        if "tiktok" in filters and not target_sources.get("tiktok", True):
            filters["tiktok"] = ""
        if "instagram" in filters and not target_sources.get("instagram", True):
            filters["instagram"] = ""
        if "social" in filters:
            if "tiktok" in filters:
                filters["social"] = str(filters.get("tiktok") or "")
            elif not target_sources.get("tiktok", True):
                filters["social"] = ""

        merged_intent["search_filters"] = filters

    return merged_intent


async def merge_intent_with_overrides(
    *,
    structured_intent: Mapping[str, Any],
    enriched_description: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Enhance intent values while preserving exact key schema from base intent."""
    base_intent = _as_mapping(structured_intent)
    if not base_intent:
        return {}

    safe_overrides = _as_mapping(overrides)
    candidate_intent: dict[str, Any] | None = None

    llm_payload = await _llm_json(
        system_prompt=_MERGE_SYSTEM_PROMPT,
        user_prompt=(
            f"structured_intent:\n{json.dumps(base_intent, ensure_ascii=True, indent=2)}\n\n"
            f"enriched_description:\n{enriched_description}\n\n"
            f"overrides:\n{json.dumps(safe_overrides, ensure_ascii=True, indent=2)}"
        ),
    )
    if isinstance(llm_payload, dict):
        candidate_intent = llm_payload

    merged = _merge_preserving_schema(base_intent, candidate_intent or base_intent)
    merged = _apply_explicit_overrides(
        merged_intent=_as_mapping(merged),
        overrides=safe_overrides,
        enriched_description=enriched_description,
    )

    return merged
