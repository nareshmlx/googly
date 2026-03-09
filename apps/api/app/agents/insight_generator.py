"""Insight generation helpers for clustering extraction and full report streaming."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import orjson
import structlog

from app.core.config import settings
from app.core.openai_client import get_openai_client
from app.core.serialization import parse_embedding as _parse_embedding
from app.models.schemas import ClusterExtraction

logger = structlog.get_logger(__name__)
_ALLOWED_TREND_SIGNALS = {"rising", "stable", "declining", "emerging", "unknown"}
_REPORT_MIN_TOKEN_BUDGET = 2600
_REPORT_MAX_EVIDENCE_SNIPPET_CHARS = 1200
_REPORT_TLDR_BULLETS_MIN = 3
_REPORT_TLDR_BULLETS_MAX = 5
_REPORT_VERIFY_MAX_CHARS = 30000
_EVIDENCE_REF_PATTERN = re.compile(r"\[E(\d+)\](?!\()")


def _coerce_metadata(raw: Any) -> dict[str, Any]:
    """Normalize chunk metadata into a dict without raising on malformed payloads."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = orjson.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    try:
        return dict(raw)
    except Exception:
        return {}


def _source_mix(chunks: list[dict[str, Any]]) -> str:
    """Return a compact source distribution string from selected chunks."""
    from collections import Counter
    counts: Counter = Counter()
    for chunk in chunks:
        source = str(chunk.get("source") or "unknown").strip() or "unknown"
        counts[source] += 1
    if not counts:
        return "unknown=0"
    ordered = counts.most_common()
    return ", ".join(f"{source}:{count}" for source, count in ordered)


def _metadata_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return first non-empty metadata value from a key preference tuple."""
    for key in keys:
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _format_evidence_block(index: int, chunk: dict[str, Any]) -> str:
    """Render one structured evidence block for model grounding."""
    metadata = chunk.get("metadata") or {}
    title = str(chunk.get("title") or "Untitled source").strip() or "Untitled source"
    source = str(chunk.get("source") or "unknown").strip() or "unknown"
    doc_id = str(chunk.get("document_id") or "").strip()
    url = _metadata_value(metadata, ("url", "link", "permalink", "video_url"))
    author = _metadata_value(metadata, ("author", "username", "channel_title", "channel"))
    published_at = _metadata_value(
        metadata,
        ("published_at", "published", "created_at", "timestamp", "date"),
    )
    snippet = str(chunk.get("content") or "").strip()[:_REPORT_MAX_EVIDENCE_SNIPPET_CHARS]
    return (
        f"<evidence id=\"E{index}\">\n"
        f"source_type: {source}\n"
        f"title: {title}\n"
        f"url: {url or 'n/a'}\n"
        f"author: {author or 'n/a'}\n"
        f"published_at: {published_at or 'n/a'}\n"
        f"document_id: {doc_id or 'n/a'}\n"
        f"snippet:\n{snippet}\n"
        "</evidence>"
    )


def _derive_report_token_budget(cluster_size: int, selected_chunk_count: int) -> int:
    """Derive adaptive max_tokens budget based on cluster complexity."""
    cap = int(settings.CLUSTER_REPORT_MAX_TOKENS)
    if cap <= 200:
        return cap

    floor = min(cap, _REPORT_MIN_TOKEN_BUDGET)
    if cap <= floor:
        return cap

    cluster_factor = min(1.0, max(0.0, cluster_size / 60.0))
    coverage_factor = min(1.0, max(0.0, selected_chunk_count / 40.0))
    complexity = (0.65 * cluster_factor) + (0.35 * coverage_factor)
    budget = int(round(floor + (cap - floor) * complexity))
    return max(floor, min(cap, budget))


def _derive_word_target(cluster_size: int, selected_chunk_count: int) -> tuple[int, int]:
    """Derive adaptive target word range for concise executive-level reporting."""
    if cluster_size >= 60 or selected_chunk_count >= 35:
        return 1100, 1600
    if cluster_size >= 30 or selected_chunk_count >= 24:
        return 900, 1300
    if cluster_size >= 12 or selected_chunk_count >= 14:
        return 750, 1050
    return 620, 900


def _prepare_report_chunks(all_cluster_chunks: list[dict], *, max_chunks: int) -> list[dict[str, Any]]:
    """Parse, score, and select report chunks from raw chunk rows."""
    parsed_chunks: list[dict[str, Any]] = []
    fallback_chunks: list[dict[str, Any]] = []
    for chunk in all_cluster_chunks:
        normalized_chunk = {
            "title": str(chunk.get("title") or "Untitled source"),
            "content": str(chunk.get("content") or ""),
            "metadata": _coerce_metadata(chunk.get("metadata")),
            "source": str(chunk.get("source") or "unknown"),
            "document_id": str(chunk.get("document_id") or ""),
        }
        fallback_chunks.append(normalized_chunk)
        emb = _parse_embedding(chunk.get("embedding"))
        if emb is None or emb.size == 0:
            continue
        parsed_chunks.append(
            {
                **normalized_chunk,
                "embedding": emb,
            }
        )
    if not parsed_chunks:
        return fallback_chunks[:max_chunks]

    embedding_matrix = np.vstack([chunk["embedding"] for chunk in parsed_chunks]).astype(np.float32)
    centroid = np.mean(embedding_matrix, axis=0)
    selected_indices = _mmr_select(
        embeddings=embedding_matrix,
        centroid=centroid,
        k=min(max_chunks, len(parsed_chunks)),
        lambda_=settings.CLUSTER_REPORT_LAMBDA,
    )
    selected_chunks = [parsed_chunks[idx] for idx in selected_indices] if selected_indices else []
    if not selected_chunks:
        selected_chunks = parsed_chunks[:max_chunks]
    return selected_chunks


def _evidence_url_map(selected_chunks: list[dict[str, Any]]) -> dict[int, str]:
    """Build 1-based evidence id -> URL map from selected chunks."""
    mapping: dict[int, str] = {}
    for index, chunk in enumerate(selected_chunks, start=1):
        metadata = chunk.get("metadata") or {}
        url = _metadata_value(metadata, ("url", "link", "permalink", "video_url"))
        if url:
            mapping[index] = url
    return mapping


def link_evidence_references(
    report_text: str,
    all_cluster_chunks: list[dict],
    *,
    max_chunks: int | None = None,
) -> str:
    """Convert plain [E#] markers into markdown links when URL is available."""
    text = str(report_text or "").strip()
    if not text:
        return text
    selected_chunks = _prepare_report_chunks(
        all_cluster_chunks,
        max_chunks=max_chunks or settings.CLUSTER_REPORT_MAX_CHUNKS,
    )
    if not selected_chunks:
        return text
    url_map = _evidence_url_map(selected_chunks)
    if not url_map:
        return text

    def _replace(match: re.Match[str]) -> str:
        evidence_id = int(match.group(1))
        url = url_map.get(evidence_id)
        if not url:
            return match.group(0)
        return f"[E{evidence_id}]({url})"

    return _EVIDENCE_REF_PATTERN.sub(_replace, text)


def _normalize_text(value: Any, *, max_len: int, fallback: str) -> str:
    """Normalize arbitrary model output into a bounded non-empty string."""
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_len].strip() or fallback


def _normalize_key_findings(value: Any) -> list[str]:
    """Normalize key findings into a clean, bounded list of strings."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip():
        items = [value]
    else:
        return []
    findings: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            findings.append(text[:500])
    return findings[: settings.CLUSTER_MAX_KEY_FINDINGS]


def _normalize_contradictions(value: Any) -> str | None:
    """Normalize contradictions into optional text."""
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item or "").strip() for item in value]
        text = "; ".join(part for part in parts if part)
    else:
        text = str(value).strip()
    return text or None


def _normalize_cluster_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce loosely-structured model JSON into ClusterExtraction shape."""
    trend_raw = str(payload.get("trend_signal") or "").strip().lower()
    trend = trend_raw if trend_raw in _ALLOWED_TREND_SIGNALS else "unknown"
    return {
        "topic_label": _normalize_text(payload.get("topic_label"), max_len=200, fallback="Untitled Insight"),
        "executive_summary": _normalize_text(
            payload.get("executive_summary"),
            max_len=300,
            fallback="Summary unavailable.",
        ),
        "key_findings": _normalize_key_findings(payload.get("key_findings")),
        "trend_signal": trend,
        "contradictions": _normalize_contradictions(payload.get("contradictions")),
    }


async def extract_cluster(
    chunks_core: list[str],
    chunks_fringe: list[str],
    model: str,
) -> ClusterExtraction | None:
    """Extract a structured cluster insight from representative core+fringe excerpts."""
    if not chunks_core and not chunks_fringe:
        return None

    client = get_openai_client()
    core_block = "\n\n".join(f"- {chunk[:1200]}" for chunk in chunks_core[:6])
    fringe_block = "\n\n".join(f"- {chunk[:1200]}" for chunk in chunks_fringe[:3])
    user_prompt = (
        "Create a compact cluster insight JSON object using only these excerpts.\n"
        "Rules:\n"
        "- executive_summary must be <= 300 characters.\n"
        f"- key_findings must contain at most {settings.CLUSTER_MAX_KEY_FINDINGS} items.\n"
        "- trend_signal must be one of: rising, stable, declining, emerging, unknown.\n"
        "- Use fringe excerpts to detect contradictions/edge cases.\n\n"
        f"<core_excerpts>\n{core_block}\n</core_excerpts>\n\n"
        f"<fringe_excerpts>\n{fringe_block}\n</fringe_excerpts>"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON generator for clustered insight extraction. "
                        "Output only a JSON object matching keys: "
                        "topic_label, executive_summary, key_findings, trend_signal, contradictions."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        payload = (response.choices[0].message.content or "").strip()
        parsed = _normalize_cluster_payload(orjson.loads(payload))
        return ClusterExtraction.model_validate(parsed)
    except Exception as exc:
        logger.warning("insight_generator.extract_cluster_failed", error=str(exc))
        return None


def _mmr_select(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    k: int,
    lambda_: float,
) -> list[int]:
    """Select diverse-yet-relevant indices using Maximal Marginal Relevance."""
    if embeddings.size == 0:
        return []
    if k <= 0:
        return []

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = embeddings / norms

    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0:
        centroid_norm = 1e-8
    centroid_unit = centroid / centroid_norm
    relevance = normalized @ centroid_unit

    n = len(normalized)
    selected: list[int] = []
    is_candidate = np.ones(n, dtype=bool)

    while len(selected) < k and np.any(is_candidate):
        candidate_indices = np.where(is_candidate)[0]
        if not selected:
            scores = relevance[candidate_indices]
        else:
            selected_matrix = normalized[selected]
            sim_to_selected = normalized[candidate_indices] @ selected_matrix.T
            max_redundancy = sim_to_selected.max(axis=1)
            scores = lambda_ * relevance[candidate_indices] - (1.0 - lambda_) * max_redundancy

        best_local = int(np.argmax(scores))
        best_idx = int(candidate_indices[best_local])
        selected.append(best_idx)
        is_candidate[best_idx] = False
    return selected


async def generate_full_report(
    insight: dict,
    all_cluster_chunks: list[dict],
) -> AsyncGenerator[str, None]:
    """Stream a detailed markdown report grounded in cluster chunk context."""
    if not all_cluster_chunks:
        return

    selected_chunks = _prepare_report_chunks(
        all_cluster_chunks,
        max_chunks=settings.CLUSTER_REPORT_MAX_CHUNKS,
    )
    if not selected_chunks:
        return

    topic_label = str(insight.get("topic_label") or "Insight")
    trend_signal = str(insight.get("trend_signal") or "unknown")
    cluster_size = int(insight.get("cluster_size") or len(all_cluster_chunks))
    contradictions = str(insight.get("contradictions") or "").strip()
    source_mix = _source_mix(selected_chunks)
    word_min, word_max = _derive_word_target(cluster_size, len(selected_chunks))
    token_budget = _derive_report_token_budget(cluster_size, len(selected_chunks))

    evidence_blocks = [
        _format_evidence_block(index, chunk)
        for index, chunk in enumerate(selected_chunks, start=1)
    ]

    contradiction_instruction = (
        "Include a 'Contradictions and Tensions' section only when conflicting evidence is present."
    )
    if not contradictions:
        contradiction_instruction = (
            "Skip 'Contradictions and Tensions' when no reliable contradiction is present."
        )

    prompt = (
        f"You are given {len(selected_chunks)} structured evidence blocks for topic '{topic_label}'.\n"
        f"Trend signal: {trend_signal}.\n"
        f"Cluster size: {cluster_size} sources. Selected source mix: {source_mix}.\n"
        f"Write an executive-ready, evidence-based markdown report with target length {word_min}-{word_max} words.\n"
        "Required outer structure only:\n"
        f"## {topic_label}\n"
        f"**Signal:** {trend_signal}\n"
        "### TL;DR\n"
        "Then 3-6 evidence-derived thematic headings chosen from the evidence (not generic templates).\n\n"
        f"{contradiction_instruction}\n"
        f"TL;DR must contain {_REPORT_TLDR_BULLETS_MIN}-{_REPORT_TLDR_BULLETS_MAX} bullets, each <= 18 words.\n"
        "For each thematic heading:\n"
        "- Write one concise synthesis paragraph.\n"
        "- Add one explicit 'Evidence:' sentence grounding the claim with 2-4 evidence IDs (e.g., [E3], [E12]).\n"
        "- Prefer concrete details (time, numbers, named actors) when present.\n"
        "Use bullets sparingly; prioritize readable narrative paragraphs.\n"
        "Use markdown links when URL exists.\n"
        "Do not invent facts beyond provided evidence.\n\n"
        "<evidence_blocks>\n"
        + "\n\n".join(evidence_blocks)
        + "\n</evidence_blocks>\n\n"
        f"Footer line: Report generated from {cluster_size} knowledge sources."
    )

    client = get_openai_client()
    try:
        stream = await client.chat.completions.create(
            model=settings.INSIGHT_REPORT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert research analyst producing detailed, grounded reports.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=token_budget,
            stream=True,
        )
        async for event in stream:
            choices = getattr(event, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            token = getattr(delta, "content", None) if delta is not None else None
            if token and isinstance(token, str):
                yield token
    except Exception as exc:
        logger.exception("insight_generator.generate_full_report_failed", error=str(exc))
        raise


async def refine_full_report(
    insight: dict[str, Any],
    draft_report: str,
    all_cluster_chunks: list[dict],
) -> str:
    """Run a verifier/rewrite pass over a draft report and return the safest final report."""
    draft = str(draft_report or "").strip()
    if not draft:
        return draft
    if not settings.INSIGHTS_REPORT_VERIFY_ENABLED:
        return draft

    selected_chunks = _prepare_report_chunks(
        all_cluster_chunks,
        max_chunks=settings.INSIGHT_REPORT_VERIFY_EVIDENCE_MAX_CHUNKS,
    )
    if not selected_chunks:
        return draft

    evidence_blocks = [
        _format_evidence_block(index, chunk)
        for index, chunk in enumerate(selected_chunks, start=1)
    ]
    topic_label = str(insight.get("topic_label") or "Insight")
    trend_signal = str(insight.get("trend_signal") or "unknown")

    prompt = (
        "Audit and improve this report for factual grounding and executive clarity.\n"
        "Return JSON only with keys: verdict, revised_report, issues.\n"
        "verdict must be one of: pass, revise.\n"
        "Rules:\n"
        "- Use only provided evidence blocks.\n"
        "- Remove or rephrase unsupported claims.\n"
        "- Keep concise executive style, avoid repetition.\n"
        "- Preserve markdown headings and citations where possible.\n"
        "- If report is already strong and grounded, set verdict=pass and revised_report to empty string.\n\n"
        f"Topic: {topic_label}\n"
        f"Signal: {trend_signal}\n\n"
        "<draft_report>\n"
        f"{draft[:_REPORT_VERIFY_MAX_CHARS]}\n"
        "</draft_report>\n\n"
        "<evidence_blocks>\n"
        + "\n\n".join(evidence_blocks)
        + "\n</evidence_blocks>"
    )

    client = get_openai_client()
    try:
        response = await client.chat.completions.create(
            model=settings.INSIGHT_REPORT_VERIFIER_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict fact-checking editor for executive research reports. "
                        "Never add facts beyond supplied evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=settings.INSIGHT_REPORT_VERIFY_MAX_TOKENS,
        )
        payload_raw = (response.choices[0].message.content or "").strip()
        payload = orjson.loads(payload_raw) if payload_raw else {}
        verdict = str(payload.get("verdict") or "pass").strip().lower()
        revised = str(payload.get("revised_report") or "").strip()

        if verdict == "revise" and revised:
            logger.info(
                "insight_generator.refine_full_report.revised",
                original_chars=len(draft),
                revised_chars=len(revised),
            )
            return revised[:_REPORT_VERIFY_MAX_CHARS].strip()
        logger.info(
            "insight_generator.refine_full_report.pass",
            draft_chars=len(draft),
        )
        return draft
    except Exception as exc:
        logger.warning("insight_generator.refine_full_report_failed", error=str(exc))
        return draft
