"""ARQ task to cluster project KB chunks into insight cards."""

from __future__ import annotations

import asyncio
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from uuid import uuid4

import numpy as np
import structlog

from app.agents.insight_generator import extract_cluster
from app.core.config import settings
from app.core.constants import RedisKeys
from app.core.metrics import (
    insights_cluster_job_duration_seconds,
    insights_cluster_jobs_total,
)
from app.core.redis import get_redis
from app.core.serialization import parse_embedding as _parse_embedding
from app.repositories import insights as insights_repo
from app.services import insights as insights_service
from app.tasks.ingest_documents_handler import _merge_ingest_status

logger = structlog.get_logger(__name__)
_KMEANS_FALLBACK_RANDOM_STATE = 42


def _hdbscan_labels(
    matrix: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> list[int]:
    """CPU-bound HDBSCAN clustering function for process executor."""
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        copy=False,
    )
    labels = clusterer.fit_predict(matrix)
    return [int(label) for label in labels.tolist()]


def _kmeans_labels(matrix: np.ndarray, n_clusters: int, random_state: int) -> list[int]:
    """Fallback clustering when HDBSCAN returns all noise labels."""
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(1024, max(32, len(matrix))),
        n_init="auto",
    )
    labels = model.fit_predict(matrix)
    return [int(label) for label in labels.tolist()]


def _split_oversized_cluster_rows(
    rows: list[dict],
    *,
    target_size: int,
    max_size: int,
    min_cluster_size: int,
) -> list[list[dict]]:
    """Split oversized clusters into smaller semantic subclusters with MiniBatchKMeans."""
    cluster_size = len(rows)
    if cluster_size <= max_size:
        return [rows]

    max_subclusters = cluster_size // max(1, min_cluster_size)
    if max_subclusters <= 1:
        return [rows]

    requested_subclusters = int(np.ceil(cluster_size / max(1, target_size)))
    start_subclusters = max(2, min(requested_subclusters, max_subclusters))
    if start_subclusters <= 1:
        return [rows]

    matrix = np.vstack([row["embedding_arr"] for row in rows]).astype(np.float32)
    normalized = _normalize_rows(matrix)

    best_groups: list[list[dict]] = [rows]
    best_max_size = cluster_size

    for subcluster_count in range(start_subclusters, max_subclusters + 1):
        labels = _kmeans_labels(normalized, subcluster_count, _KMEANS_FALLBACK_RANDOM_STATE)

        grouped: dict[int, list[dict]] = {}
        for idx, label in enumerate(labels):
            grouped.setdefault(int(label), []).append(rows[idx])

        valid_groups = [group for group in grouped.values() if len(group) >= min_cluster_size]
        tiny_groups = [group for group in grouped.values() if len(group) < min_cluster_size]
        if not valid_groups:
            continue

        if tiny_groups:
            for tiny_group in tiny_groups:
                for row in tiny_group:
                    smallest_group_idx = min(
                        range(len(valid_groups)),
                        key=lambda i: len(valid_groups[i]),
                    )
                    valid_groups[smallest_group_idx].append(row)

        candidate_groups = [group for group in valid_groups if group]
        if not candidate_groups:
            continue

        candidate_max_size = max(len(group) for group in candidate_groups)
        if candidate_max_size < best_max_size:
            best_groups = candidate_groups
            best_max_size = candidate_max_size

        if candidate_max_size <= max_size:
            return candidate_groups

    return best_groups


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so euclidean distance approximates cosine distance."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms


def _cosine_scores(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Return cosine similarity of every vector against centroid."""
    vector_norms = np.linalg.norm(vectors, axis=1)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0:
        centroid_norm = 1e-8
    safe_norms = np.where(vector_norms == 0, 1e-8, vector_norms)
    return (vectors @ centroid) / (safe_norms * centroid_norm)


def _normalize_topic_label(label: str) -> str:
    """Normalize topic labels for deterministic duplicate detection."""
    lowered = "".join(ch.lower() if ch.isalnum() else " " for ch in str(label or ""))
    return " ".join(lowered.split())


def _ordered_unique(values: list[str]) -> list[str]:
    """Return stable-order unique list of non-empty string values."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _cluster_raw_sources(cluster_rows: list[dict]) -> set[str]:
    """Return the raw source names represented in one cluster."""
    return {
        str(row.get("source") or "").strip()
        for row in cluster_rows
        if str(row.get("source") or "").strip()
    }


def _source_family(source_name: str) -> str:
    """Map raw source names to broad source families."""
    source = str(source_name or "").strip().lower()
    if source.startswith("social_"):
        return "social"
    if source in {"paper", "search"}:
        return "research"
    if source in {"news", "patent"}:
        return source
    return source or "unknown"


def _to_non_negative_int(value: object) -> int:
    """Coerce values to non-negative ints for count arithmetic."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return max(0, int(float(text)))
    except (TypeError, ValueError):
        return 0


def _dominant_source_family(source_type_counts: dict[str, int]) -> str:
    """Return dominant source family from per-source counts."""
    family_counts: Counter[str] = Counter()
    for source_name, raw_count in (source_type_counts or {}).items():
        count = _to_non_negative_int(raw_count)
        if count > 0:
            family_counts[_source_family(source_name)] += count
    if not family_counts:
        return "unknown"
    return family_counts.most_common(1)[0][0]


def _doc_jaccard_similarity(left_ids: list[str], right_ids: list[str]) -> float:
    """Return Jaccard similarity for source-document id sets."""
    left = {str(doc_id or "").strip() for doc_id in left_ids if str(doc_id or "").strip()}
    right = {str(doc_id or "").strip() for doc_id in right_ids if str(doc_id or "").strip()}
    if not left or not right:
        return 0.0
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


def _text_token_set(value: str) -> set[str]:
    """Tokenize free text into a normalized set for coarse similarity checks."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(value or ""))
    return {token for token in cleaned.split() if len(token) >= 3}


def _summary_jaccard_similarity(left_row: dict, right_row: dict) -> float:
    """Return Jaccard similarity across summary/finding text for duplicate-title safety."""
    left_text = " ".join(
        [
            str(left_row.get("executive_summary") or ""),
            " ".join(str(item or "") for item in list(left_row.get("key_findings") or [])),
        ]
    )
    right_text = " ".join(
        [
            str(right_row.get("executive_summary") or ""),
            " ".join(str(item or "") for item in list(right_row.get("key_findings") or [])),
        ]
    )
    left_tokens = _text_token_set(left_text)
    right_tokens = _text_token_set(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    union_size = len(left_tokens | right_tokens)
    if union_size == 0:
        return 0.0
    return len(left_tokens & right_tokens) / union_size


def _merge_insight_rows(primary: dict, duplicate: dict) -> dict:
    """Merge duplicate insight rows while preserving best available fields."""
    merged = dict(primary)
    primary_size = int(primary.get("cluster_size") or 0)
    duplicate_size = int(duplicate.get("cluster_size") or 0)

    merged["cluster_size"] = primary_size + duplicate_size
    merged["source_doc_ids"] = _ordered_unique(
        list(primary.get("source_doc_ids") or []) + list(duplicate.get("source_doc_ids") or [])
    )
    merged["chunk_ids"] = _ordered_unique(
        list(primary.get("chunk_ids") or []) + list(duplicate.get("chunk_ids") or [])
    )

    source_type_counts: Counter[str] = Counter()
    for row in (primary, duplicate):
        for source_name, raw_count in (row.get("source_type_counts") or {}).items():
            source_type_counts[str(source_name)] += _to_non_negative_int(raw_count)
    merged["source_type_counts"] = dict(source_type_counts)

    key_findings: list[str] = []
    for finding in list(primary.get("key_findings") or []) + list(duplicate.get("key_findings") or []):
        text = str(finding or "").strip()
        if text and text not in key_findings:
            key_findings.append(text)
        if len(key_findings) >= settings.CLUSTER_MAX_KEY_FINDINGS:
            break
    merged["key_findings"] = key_findings

    if duplicate_size > primary_size:
        for field in ("topic_label", "executive_summary", "trend_signal", "contradictions"):
            if duplicate.get(field):
                merged[field] = duplicate.get(field)
    elif not merged.get("contradictions") and duplicate.get("contradictions"):
        merged["contradictions"] = duplicate.get("contradictions")

    return merged


def _topic_disambiguation_suffix(row: dict) -> str:
    """Return short display suffix for duplicated topic labels."""
    family = _dominant_source_family(dict(row.get("source_type_counts") or {}))
    if family == "social":
        return "Social"
    if family == "research":
        return "Research"
    if family == "patent":
        return "Patents"
    if family == "news":
        return "News"
    return "Mixed"


def _rank_cluster_ids_for_output(clusters: dict[int, list[dict]], *, max_clusters: int) -> list[int]:
    """Rank output clusters while reserving space for target source types when available."""
    if not clusters or max_clusters <= 0:
        return []

    def _distinct_doc_count(cluster_rows_local: list[dict]) -> int:
        return len(
            {
                str(row.get("document_id") or "").strip()
                for row in cluster_rows_local
                if str(row.get("document_id") or "").strip()
            }
        )

    def _distinct_source_count(cluster_rows_local: list[dict]) -> int:
        return len(_cluster_raw_sources(cluster_rows_local))

    ranked_cluster_ids = sorted(
        clusters.keys(),
        key=lambda cluster_id: (
            _distinct_doc_count(clusters[cluster_id]),
            _distinct_source_count(clusters[cluster_id]),
            len(clusters[cluster_id]),
        ),
        reverse=True,
    )

    selected: list[int] = []
    reserved_sources = ("social_tiktok", "paper", "patent")
    reserved_slot_budget = min(max_clusters, len(reserved_sources))
    for source_name in reserved_sources:
        if len(selected) >= reserved_slot_budget:
            break
        match_id = next(
            (
                cluster_id
                for cluster_id in ranked_cluster_ids
                if cluster_id not in selected and source_name in _cluster_raw_sources(clusters[cluster_id])
            ),
            None,
        )
        if match_id is not None:
            selected.append(match_id)

    for cluster_id in ranked_cluster_ids:
        if cluster_id in selected:
            continue
        selected.append(cluster_id)
        if len(selected) >= max_clusters:
            break

    return selected[:max_clusters]


def _source_counts_from_rows(cluster_rows: list[dict], *, unique_documents: bool) -> dict[str, int]:
    """Count source participation across cluster rows, optionally deduplicated by document."""
    counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    for row in cluster_rows:
        source_name = str(row.get("source") or "unknown").strip() or "unknown"
        document_id = str(row.get("document_id") or "").strip()
        if unique_documents:
            pair = (source_name, document_id)
            if document_id and pair in seen_pairs:
                continue
            if document_id:
                seen_pairs.add(pair)
        counts[source_name] += 1
    return dict(counts)


def _build_cluster_diagnostics(
    *,
    candidate_rows: list[dict],
    ranked_cluster_ids: list[int],
    clusters: dict[int, list[dict]],
    insight_rows: list[dict],
    parse_failed_clusters: int,
    fallback_cluster_count: int,
    noise_chunk_count: int,
    noise_reassigned: int,
) -> dict[str, object]:
    """Build persisted diagnostics explaining how sources flowed through clustering."""
    candidate_chunk_counts = _source_counts_from_rows(candidate_rows, unique_documents=False)
    candidate_doc_counts = _source_counts_from_rows(candidate_rows, unique_documents=True)

    selected_cluster_counts: Counter[str] = Counter()
    selected_doc_counts: Counter[str] = Counter()
    for cluster_id in ranked_cluster_ids:
        cluster_rows = clusters.get(cluster_id) or []
        for source_name in _cluster_raw_sources(cluster_rows):
            selected_cluster_counts[source_name] += 1
    for row in insight_rows:
        for source_name, raw_count in dict(row.get("source_type_counts") or {}).items():
            selected_doc_counts[str(source_name)] += _to_non_negative_int(raw_count)

    unclustered_doc_counts = {
        source_name: max(0, int(candidate_doc_counts.get(source_name, 0)) - int(selected_doc_counts.get(source_name, 0)))
        for source_name in set(candidate_doc_counts) | set(selected_doc_counts)
    }

    return {
        "status": "success",
        "candidate_chunk_counts": candidate_chunk_counts,
        "candidate_doc_counts": candidate_doc_counts,
        "selected_cluster_counts": dict(selected_cluster_counts),
        "selected_doc_counts": dict(selected_doc_counts),
        "unclustered_doc_counts": unclustered_doc_counts,
        "ranked_cluster_count": len(ranked_cluster_ids),
        "persisted_insight_count": len(insight_rows),
        "parse_failed_clusters": int(parse_failed_clusters),
        "fallback_cluster_count": int(fallback_cluster_count),
        "noise_chunk_count": int(noise_chunk_count),
        "noise_reassigned": int(noise_reassigned),
    }


def _consolidate_and_disambiguate_topics(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Merge safe duplicate titles and disambiguate any remaining title collisions."""
    if not rows:
        return rows, 0, 0

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = _normalize_topic_label(str(row.get("topic_label") or ""))
        grouped.setdefault(key, []).append(row)

    merged_rows: list[dict] = []
    merged_duplicate_count = 0

    for group in grouped.values():
        if len(group) == 1:
            merged_rows.extend(group)
            continue

        survivors: list[dict] = []
        ordered_group = sorted(group, key=lambda r: int(r.get("cluster_size") or 0), reverse=True)
        for candidate in ordered_group:
            merged = False
            if settings.CLUSTER_MERGE_DUPLICATE_TITLES:
                for idx, existing in enumerate(survivors):
                    overlap = _doc_jaccard_similarity(
                        list(existing.get("source_doc_ids") or []),
                        list(candidate.get("source_doc_ids") or []),
                    )
                    summary_similarity = _summary_jaccard_similarity(existing, candidate)
                    if (
                        overlap >= settings.CLUSTER_DUPLICATE_TITLE_MIN_DOC_JACCARD
                        or summary_similarity >= settings.CLUSTER_DUPLICATE_TITLE_MIN_SUMMARY_JACCARD
                    ):
                        survivors[idx] = _merge_insight_rows(existing, candidate)
                        merged_duplicate_count += 1
                        merged = True
                        break
            if not merged:
                survivors.append(candidate)
        merged_rows.extend(survivors)

    renamed_duplicate_count = 0
    renamed_groups: dict[str, list[dict]] = {}
    for row in merged_rows:
        key = _normalize_topic_label(str(row.get("topic_label") or ""))
        renamed_groups.setdefault(key, []).append(row)

    for group in renamed_groups.values():
        if len(group) <= 1:
            continue
        qualifier_counts: Counter[str] = Counter()
        for row in sorted(group, key=lambda r: int(r.get("cluster_size") or 0), reverse=True):
            base_label = str(row.get("topic_label") or "Untitled Insight").strip() or "Untitled Insight"
            qualifier = _topic_disambiguation_suffix(row)
            qualifier_counts[qualifier] += 1
            qualifier_with_index = (
                qualifier if qualifier_counts[qualifier] == 1 else f"{qualifier} {qualifier_counts[qualifier]}"
            )
            new_label = f"{base_label} ({qualifier_with_index})"
            if len(new_label) > 200:
                new_label = f"{new_label[:197].rstrip()}..."
            if new_label != base_label:
                row["topic_label"] = new_label
                renamed_duplicate_count += 1

    ordered_rows = sorted(
        merged_rows,
        key=lambda row: int(row.get("cluster_size") or 0),
        reverse=True,
    )
    return ordered_rows, merged_duplicate_count, renamed_duplicate_count


async def _renew_cluster_lock(project_id: str, stop_event: asyncio.Event) -> None:
    """Renew cluster lock TTL until stop event is set."""
    redis = await get_redis()
    lock_key = RedisKeys.CLUSTER_LOCK.format(project_id=project_id)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=settings.CLUSTER_LOCK_HEARTBEAT_SECONDS,
            )
            break
        except TimeoutError:
            try:
                await redis.expire(lock_key, settings.CLUSTER_LOCK_TTL)
            except Exception as exc:
                logger.warning(
                    "cluster_project.lock_heartbeat_failed",
                    project_id=project_id,
                    error=str(exc),
                )


async def cluster_project(ctx: dict, project_id: str) -> None:
    """Cluster project KB chunks and atomically replace insight rows."""
    started_at = perf_counter()
    if not settings.INSIGHTS_ENABLED:
        insights_cluster_jobs_total.labels(status="skipped_disabled").inc()
        logger.info("cluster_project.skipped.disabled", project_id=project_id)
        return

    redis = await get_redis()
    lock_key = RedisKeys.CLUSTER_LOCK.format(project_id=project_id)
    enqueue_lock_key = RedisKeys.CLUSTER_ENQUEUE_LOCK.format(project_id=project_id)
    dirty_key = RedisKeys.CLUSTER_DIRTY.format(project_id=project_id)
    acquired = await redis.set(lock_key, "1", ex=settings.CLUSTER_LOCK_TTL, nx=True)
    if not acquired:
        insights_cluster_jobs_total.labels(status="skipped_lock").inc()
        logger.debug("cluster_project.skipped.lock_exists", project_id=project_id)
        return
    insights_cluster_jobs_total.labels(status="started").inc()

    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(_renew_cluster_lock(project_id, stop_heartbeat))

    try:
        chunk_rows = await insights_repo.get_cluster_candidate_chunks_for_service(
            project_id,
            limit=settings.CLUSTER_MAX_CHUNKS,
            max_per_document=settings.CLUSTER_MAX_CHUNKS_PER_DOCUMENT,
        )
        if len(chunk_rows) < 5:
            insights_cluster_jobs_total.labels(status="skipped_small").inc()
            await _merge_ingest_status(
                redis,
                project_id,
                cluster_diagnostics={
                    "status": "skipped_not_enough_chunks",
                    "candidate_chunk_counts": _source_counts_from_rows(chunk_rows, unique_documents=False),
                    "candidate_doc_counts": _source_counts_from_rows(chunk_rows, unique_documents=True),
                    "persisted_insight_count": 0,
                },
            )
            logger.info(
                "cluster_project.skipped.not_enough_chunks",
                project_id=project_id,
                chunk_count=len(chunk_rows),
            )
            return

        for row in chunk_rows:
            row["embedding_arr"] = _parse_embedding(str(row["embedding"]))
        chunk_rows = [row for row in chunk_rows if row["embedding_arr"] is not None]

        if len(chunk_rows) < settings.CLUSTER_MIN_CLUSTER_SIZE:
            insights_cluster_jobs_total.labels(status="skipped_small").inc()
            await _merge_ingest_status(
                redis,
                project_id,
                cluster_diagnostics={
                    "status": "skipped_not_enough_valid_embeddings",
                    "candidate_chunk_counts": _source_counts_from_rows(chunk_rows, unique_documents=False),
                    "candidate_doc_counts": _source_counts_from_rows(chunk_rows, unique_documents=True),
                    "persisted_insight_count": 0,
                },
            )
            logger.info(
                "cluster_project.skipped.not_enough_valid_embeddings",
                project_id=project_id,
                chunk_count=len(chunk_rows),
            )
            return

        embedding_matrix = np.vstack([row["embedding_arr"] for row in chunk_rows]).astype(np.float32)
        normalized_matrix = _normalize_rows(embedding_matrix)

        loop = asyncio.get_running_loop()
        try:
            mp_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=settings.CLUSTER_CPU_WORKERS,
                mp_context=mp_context,
            ) as executor:
                labels = await loop.run_in_executor(
                    executor,
                    _hdbscan_labels,
                    normalized_matrix,
                    settings.CLUSTER_MIN_CLUSTER_SIZE,
                    settings.CLUSTER_MIN_SAMPLES,
                )
        except Exception as exc:
            logger.warning(
                "cluster_project.process_pool_failed_fallback_thread",
                project_id=project_id,
                error=str(exc),
            )
            labels = await asyncio.to_thread(
                _hdbscan_labels,
                normalized_matrix,
                settings.CLUSTER_MIN_CLUSTER_SIZE,
                settings.CLUSTER_MIN_SAMPLES,
            )

        if all(label == -1 for label in labels):
            sample_count = len(chunk_rows)
            max_reasonable_clusters = max(
                2,
                min(
                    settings.CLUSTER_MAX_CLUSTERS,
                    sample_count // max(1, settings.CLUSTER_MIN_CLUSTER_SIZE),
                ),
            )
            estimated_clusters = max(2, int(round(float(np.sqrt(sample_count)))))
            fallback_clusters = max(2, min(estimated_clusters, max_reasonable_clusters))
            labels = await asyncio.to_thread(
                _kmeans_labels,
                normalized_matrix,
                fallback_clusters,
                _KMEANS_FALLBACK_RANDOM_STATE,
            )
            logger.info(
                "cluster_project.all_noise_fallback_kmeans",
                project_id=project_id,
                sample_count=sample_count,
                fallback_clusters=fallback_clusters,
            )

        cluster_indices: dict[int, list[int]] = {}
        noise_indices: list[int] = []
        for idx, label in enumerate(labels):
            if label == -1:
                noise_indices.append(idx)
                continue
            cluster_indices.setdefault(int(label), []).append(idx)

        noise_reassigned = 0
        if len(cluster_indices) > 1 and noise_indices:
            centroids: dict[int, np.ndarray] = {}
            for cluster_id, member_indices in cluster_indices.items():
                centroid = np.mean(normalized_matrix[member_indices], axis=0)
                norm = float(np.linalg.norm(centroid))
                centroids[cluster_id] = centroid if norm == 0 else (centroid / norm)

            for noise_idx in noise_indices:
                vector = normalized_matrix[noise_idx]
                best_cluster_id: int | None = None
                best_similarity = -1.0
                for cluster_id, centroid in centroids.items():
                    similarity = float(vector @ centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster_id = cluster_id
                if (
                    best_cluster_id is not None
                    and best_similarity >= settings.CLUSTER_NOISE_ASSIGNMENT_MIN_SIMILARITY
                    and len(cluster_indices[best_cluster_id]) < settings.CLUSTER_MAX_CLUSTER_SIZE
                ):
                    cluster_indices[best_cluster_id].append(noise_idx)
                    noise_reassigned += 1

        base_clusters: dict[int, list[dict]] = {
            cluster_id: [chunk_rows[idx] for idx in member_indices]
            for cluster_id, member_indices in cluster_indices.items()
            if member_indices
        }

        split_clusters: dict[int, list[dict]] = {}
        split_cluster_count = 0
        next_cluster_id = 0
        for rows in base_clusters.values():
            split_groups = _split_oversized_cluster_rows(
                rows,
                target_size=settings.CLUSTER_TARGET_CLUSTER_SIZE,
                max_size=settings.CLUSTER_MAX_CLUSTER_SIZE,
                min_cluster_size=settings.CLUSTER_MIN_CLUSTER_SIZE,
            )
            split_cluster_count += max(0, len(split_groups) - 1)
            for group in split_groups:
                split_clusters[next_cluster_id] = group
                next_cluster_id += 1

        clusters = split_clusters
        if not clusters:
            insights_cluster_jobs_total.labels(status="skipped_empty").inc()
            await _merge_ingest_status(
                redis,
                project_id,
                cluster_diagnostics={
                    "status": "skipped_no_clusters",
                    "candidate_chunk_counts": _source_counts_from_rows(chunk_rows, unique_documents=False),
                    "candidate_doc_counts": _source_counts_from_rows(chunk_rows, unique_documents=True),
                    "persisted_insight_count": 0,
                },
            )
            logger.info("cluster_project.no_clusters", project_id=project_id)
            return

        ranked_cluster_ids = _rank_cluster_ids_for_output(
            clusters,
            max_clusters=settings.CLUSTER_MAX_CLUSTERS,
        )

        semaphore = asyncio.Semaphore(settings.CLUSTER_EXTRACTION_MAX_CONCURRENCY)

        async def _extract_cluster_row(cluster_rows_local: list[dict]) -> dict | None:
            vectors = np.vstack([row["embedding_arr"] for row in cluster_rows_local]).astype(np.float32)
            centroid = np.mean(vectors, axis=0)
            sims = _cosine_scores(vectors, centroid)
            ranked_desc = sorted(range(len(cluster_rows_local)), key=lambda i: float(sims[i]), reverse=True)
            core_idx = ranked_desc[:6]

            fringe_candidates = sorted(
                (i for i in range(len(cluster_rows_local)) if float(sims[i]) >= 0.25),
                key=lambda i: float(sims[i]),
            )
            fringe_idx: list[int] = []
            for i in fringe_candidates:
                if i in core_idx:
                    continue
                fringe_idx.append(i)
                if len(fringe_idx) == 3:
                    break

            selected_idx: list[int] = []
            for i in core_idx + fringe_idx:
                if i not in selected_idx:
                    selected_idx.append(i)

            core_texts = [
                f"{cluster_rows_local[i].get('title') or 'Untitled'}\n{cluster_rows_local[i].get('content') or ''}"
                for i in core_idx
            ]
            fringe_texts = [
                f"{cluster_rows_local[i].get('title') or 'Untitled'}\n{cluster_rows_local[i].get('content') or ''}"
                for i in fringe_idx
            ]

            async with semaphore:
                extraction = await extract_cluster(core_texts, fringe_texts, settings.SUMMARY_MODEL)
            if extraction is None:
                async with semaphore:
                    extraction = await extract_cluster(core_texts[:3], [], settings.SUMMARY_MODEL)

            fallback_used = False
            if extraction is None:
                fallback_used = True
                lead_title = str(cluster_rows_local[ranked_desc[0]].get("title") or "").strip()
                lead_content = str(cluster_rows_local[ranked_desc[0]].get("content") or "").strip()
                lead_content = " ".join(lead_content.split())
                if len(lead_content) > 300:
                    lead_content = f"{lead_content[:297].rstrip()}..."
                extraction_topic = lead_title[:200] or "Untitled Insight"
                extraction_summary = lead_content or "Summary unavailable for this source cluster."
                extraction_findings: list[str] = []
                extraction_trend = "unknown"
                extraction_contradictions = None
            else:
                extraction_topic = extraction.topic_label
                extraction_summary = extraction.executive_summary
                extraction_findings = extraction.key_findings[: settings.CLUSTER_MAX_KEY_FINDINGS]
                extraction_trend = extraction.trend_signal
                extraction_contradictions = extraction.contradictions

            source_doc_ids: list[str] = []
            seen_doc_ids: set[str] = set()
            source_type_counts: Counter[str] = Counter()
            for row in cluster_rows_local:
                doc_id = str(row.get("document_id") or "").strip()
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    source_doc_ids.append(doc_id)
                    source_type_counts[str(row.get("source") or "unknown")] += 1

            representative_chunk_ids = [str(cluster_rows_local[i]["id"]) for i in selected_idx]
            return {
                "id": str(uuid4()),
                "topic_label": extraction_topic,
                "executive_summary": extraction_summary,
                "key_findings": extraction_findings,
                "trend_signal": extraction_trend,
                "contradictions": extraction_contradictions,
                "chunk_ids": representative_chunk_ids,
                "source_doc_ids": source_doc_ids,
                "cluster_size": len(cluster_rows_local),
                "full_report": None,
                "full_report_status": "pending",
                "source_type_counts": dict(source_type_counts),
                "__fallback_used": fallback_used,
            }

        extracted_rows = await asyncio.gather(
            *[_extract_cluster_row(clusters[cluster_id]) for cluster_id in ranked_cluster_ids],
            return_exceptions=True,
        )
        # Treat exceptions as parse failures (same as None return)
        parse_failed_clusters = sum(
            1 for row in extracted_rows if row is None or isinstance(row, BaseException)
        )
        insight_rows = [
            row for row in extracted_rows if row is not None and not isinstance(row, BaseException)
        ]
        fallback_cluster_count = 0
        for row in insight_rows:
            if bool(row.pop("__fallback_used", False)):
                fallback_cluster_count += 1
        (
            insight_rows,
            merged_duplicate_title_count,
            renamed_duplicate_title_count,
        ) = _consolidate_and_disambiguate_topics(insight_rows)
        if not insight_rows:
            insights_cluster_jobs_total.labels(status="skipped_empty").inc()
            await _merge_ingest_status(
                redis,
                project_id,
                cluster_diagnostics={
                    "status": "skipped_empty_extraction",
                    "candidate_chunk_counts": _source_counts_from_rows(chunk_rows, unique_documents=False),
                    "candidate_doc_counts": _source_counts_from_rows(chunk_rows, unique_documents=True),
                    "ranked_cluster_count": len(ranked_cluster_ids),
                    "parse_failed_clusters": int(parse_failed_clusters),
                    "persisted_insight_count": 0,
                },
            )
            logger.warning(
                "cluster_project.skipped.replace_due_to_empty_extraction",
                project_id=project_id,
                cluster_count=len(ranked_cluster_ids),
                parse_failed_clusters=parse_failed_clusters,
            )
            return

        source_doc_ids = [
            doc_id
            for row in insight_rows
            for doc_id in row.get("source_doc_ids", [])
            if str(doc_id).strip()
        ]
        if source_doc_ids:
            await insights_service.ensure_source_doc_summaries(project_id, source_doc_ids)

        await insights_repo.bulk_replace_insights_for_service(project_id, insight_rows)
        await _merge_ingest_status(
            redis,
            project_id,
            cluster_diagnostics=_build_cluster_diagnostics(
                candidate_rows=chunk_rows,
                ranked_cluster_ids=ranked_cluster_ids,
                clusters=clusters,
                insight_rows=insight_rows,
                parse_failed_clusters=parse_failed_clusters,
                fallback_cluster_count=fallback_cluster_count,
                noise_chunk_count=len(noise_indices),
                noise_reassigned=noise_reassigned,
            ),
        )

        await redis.delete(RedisKeys.INSIGHTS_CACHE.format(project_id=project_id))
        await redis.incr(RedisKeys.PROJECT_CACHE_VERSION.format(project_id=project_id))

        insights_cluster_jobs_total.labels(status="success").inc()
        duration_seconds = perf_counter() - started_at
        insights_cluster_job_duration_seconds.observe(duration_seconds)
        logger.info(
            "cluster_project.done",
            project_id=project_id,
            clusters_inserted=len(insight_rows),
            parse_failed_clusters=parse_failed_clusters,
            fallback_cluster_count=fallback_cluster_count,
            total_chunk_count=len(chunk_rows),
            base_cluster_count=len(base_clusters),
            split_cluster_count=split_cluster_count,
            clustered_chunk_count=sum(len(rows) for rows in clusters.values()),
            noise_chunk_count=len(noise_indices),
            noise_reassigned=noise_reassigned,
            merged_duplicate_title_count=merged_duplicate_title_count,
            renamed_duplicate_title_count=renamed_duplicate_title_count,
            duration_seconds=round(duration_seconds, 3),
            history_replaced=True,
        )
    except Exception as exc:
        insights_cluster_jobs_total.labels(status="failed").inc()
        duration_seconds = perf_counter() - started_at
        insights_cluster_job_duration_seconds.observe(duration_seconds)
        await _merge_ingest_status(
            redis,
            project_id,
            cluster_diagnostics={
                "status": "failed",
                "error": str(exc),
            },
        )
        logger.exception(
            "cluster_project.failed",
            project_id=project_id,
            error=str(exc),
            duration_seconds=round(duration_seconds, 3),
        )
        raise
    finally:
        stop_heartbeat.set()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await redis.delete(lock_key)
        await redis.delete(enqueue_lock_key)
        rerun_requested = bool(await redis.delete(dirty_key))
        arq_pool = ctx.get("redis")
        if rerun_requested and arq_pool is not None:
            try:
                await arq_pool.enqueue_job("cluster_project_task", project_id)
                logger.info("cluster_project.reenqueued_dirty_project", project_id=project_id)
            except Exception as exc:
                logger.warning(
                    "cluster_project.reenqueue_failed",
                    project_id=project_id,
                    error=str(exc),
                )


async def cluster_project_task(ctx: dict, project_id: str) -> None:
    """ARQ task alias for project clustering.

    Uses a stable dedicated task name to avoid resolver collisions with legacy
    queue entries while preserving the same clustering behavior.
    """
    await cluster_project(ctx, project_id)
