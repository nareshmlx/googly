"""Relevance filtering logic for ingestion (Embedding, LLM judging)."""

import asyncio

import numpy as np
import structlog

from app.core.config import settings
from app.kb.embedder import embed_texts
from app.services.llm import chat_completion
from app.tasks.ingest_utils import (
    _relevance_item_text,
    _required_must_match_count,
    _tokenize,
)

logger = structlog.get_logger(__name__)


async def _filter_relevance(
    items: list[dict],
    intent_text: str,
    source: str,
    redis,
    must_match_terms: list[str] | None = None,
    social_match_terms: list[str] | None = None,
    intent_embedding: list[float] | None = None,
) -> list[dict]:
    """Apply two-stage relevance filtering with fail-open behavior."""
    if not items:
        return []

    # Stage 1: Embedding-based similarity filter
    candidates = await _filter_stage1_embedding(
        items,
        intent_text,
        redis,
        source=source,
        intent_embedding=intent_embedding,
    )
    if not candidates:
        return []

    # Stage 2: LLM-based qualitative judging
    if source in ("paper", "patent"):
        return await _filter_stage2_llm(
            candidates,
            intent_text,
            source,
            must_match_terms=must_match_terms,
        )
    elif source in (
        "social_instagram",
        "social_tiktok",
        "social_youtube",
        "social_x",
        "social_reddit",
    ):
        return await _filter_stage2_llm_social(
            candidates,
            intent_text=intent_text,
            source=source,
            must_match_terms=must_match_terms,
        )

    return candidates


async def _filter_stage1_embedding(
    items: list[dict],
    intent_text: str,
    redis,
    source: str | None = None,
    intent_embedding: list[float] | None = None,
) -> list[dict]:
    """Drop items in the bottom 40th percentile of intent cosine similarity."""
    if not items:
        return []

    try:
        texts = [_relevance_item_text(it) for it in items]

        # Win #5: Reuse intent_embedding if provided
        if intent_embedding is None:
            all_embs = await embed_texts([intent_text] + texts)
            intent_emb = np.array(all_embs[0])
            doc_embs = [np.array(e) for e in all_embs[1:]]
        else:
            intent_emb = np.array(intent_embedding)
            doc_embs = [np.array(e) for e in await embed_texts(texts)]

        scores = []
        for i, emb in enumerate(doc_embs):
            norm_intent = np.linalg.norm(intent_emb)
            norm_emb = np.linalg.norm(emb)
            if norm_intent == 0 or norm_emb == 0:
                score = 0.0
            else:
                score = float(np.dot(intent_emb, emb) / (norm_intent * norm_emb))
            scores.append((score, items[i]))

        if not scores:
            return items

        # Drop bottom 20% (keep top 80%) - reduced from 40% for better recall
        scores.sort(key=lambda x: x[0], reverse=True)
        keep_count = max(1, int(len(scores) * 0.8))
        kept = [it for _, it in scores[:keep_count]]

        logger.info(
            "filter_stage1_embedding.complete",
            source=source,
            candidates=len(items),
            kept=len(kept),
            filtered_out=len(items) - len(kept),
            min_score=scores[-1][0] if scores else 0,
            median_score=scores[len(scores) // 2][0] if scores else 0,
            threshold_score=scores[keep_count - 1][0] if keep_count > 0 else 0,
        )

        return kept
    except Exception:
        logger.exception("filter_stage1.failed", source=source)
        return items


async def _filter_stage2_llm(
    items: list[dict],
    intent_text: str,
    source: str,
    must_match_terms: list[str] | None = None,
) -> list[dict]:
    """Use batched GPT-4o-mini relevance judging for papers and patents."""
    if not items:
        return []

    must_terms = set(must_match_terms or [])
    required_count = _required_must_match_count(must_terms)

    async def _judge(item: dict) -> bool:
        content = _relevance_item_text(item)
        if must_terms:
            tokens = _tokenize(content)
            matches = sum(1 for t in must_terms if t in tokens)
            if matches < required_count:
                return False

        prompt = (
            f"You are a research assistant judging the relevance of a {source}.\n"
            f"Target Context: {intent_text}\n\n"
            f"Candidate Content: {content[:1000]}\n\n"
            "Is this extremely relevant to the target context? Reply exactly with YES or NO."
        )
        try:
            resp = await chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=settings.ANALYZER_MODEL,
                temperature=0,
                max_tokens=5,
            )
            return "YES" in str(resp).upper()
        except Exception:
            return True

    results = await asyncio.gather(*[_judge(it) for it in items])
    return [it for it, keep in zip(items, results, strict=False) if keep]


async def _filter_stage2_llm_social(
    items: list[dict],
    *,
    intent_text: str,
    source: str,
    must_match_terms: list[str] | None = None,
) -> list[dict]:
    """Use batched GPT-4o-mini relevance judging for social items."""
    if not items:
        return []

    must_terms = set(must_match_terms or [])
    required_count = _required_must_match_count(must_terms)

    term_filtered = 0
    llm_filtered = 0

    async def _judge_social(item: dict) -> tuple[bool, str]:
        nonlocal term_filtered, llm_filtered
        content = _relevance_item_text(item)

        # First check LLM judgment (most important for social)
        prompt = (
            f"Judge if this {source} post is relevant to: {intent_text}\n"
            f"Post Content: {content[:800]}\n"
            "Reply with YES if it's high quality and relevant, NO otherwise."
        )
        try:
            resp = await chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=5,
            )
            is_relevant = "YES" in str(resp).upper()

            # If LLM says YES, keep it (even if no term matches)
            if is_relevant:
                return True, "llm_approved"

            # If LLM says NO, check term matches as fallback
            if must_terms:
                tokens = _tokenize(content)
                matches = sum(1 for t in must_terms if t in tokens)
                if matches >= required_count:
                    # Has required terms even though LLM rejected - keep it
                    return True, "term_match_override"
                term_filtered += 1

            llm_filtered += 1
            return False, "llm_and_terms_rejected"
        except Exception:
            # On LLM error, fall back to term matching if available
            if must_terms:
                tokens = _tokenize(content)
                matches = sum(1 for t in must_terms if t in tokens)
                if matches >= required_count:
                    return True, "llm_error_term_fallback"
                term_filtered += 1
                return False, "llm_error_no_terms"
            return True, "llm_error_fail_open"

    results = await asyncio.gather(*[_judge_social(it) for it in items])
    kept = [it for it, (keep, _reason) in zip(items, results, strict=False) if keep]

    logger.info(
        "filter_stage2_social.complete",
        source=source,
        candidates=len(items),
        kept=len(kept),
        term_filtered=term_filtered,
        llm_filtered=llm_filtered,
        must_terms_count=len(must_terms),
    )

    return kept
