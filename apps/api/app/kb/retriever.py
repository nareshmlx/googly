"""KB Retriever — pgvector cosine similarity search with project pre-filter and hot cache.

Design decisions:
- Pre-filter by project_id(s) BEFORE the vector scan. Without this, pgvector scans
  all rows and then filters — catastrophic at 10k users with millions of chunks.
  With the pre-filter index (ix_kc_project_id), Postgres narrows the candidate set
  first, then does the KNN scan on the reduced set.
- RLS via SET app.accessible_projects per connection — enforces multi-tenant isolation
  even if the caller forgets to pass project_ids (belt-and-suspenders).
- Redis hot cache: frequently-retrieved chunks are cached for 1 hr. Key is
  kb_hot:{project_id_hash}:{query_hash} so multi-project searches get their own slot.
- Returns None (not empty list) when best score < KB_SCORE_THRESHOLD — the caller
  (agent) interprets None as "trigger fallback search", not "no results".
- Per-source quotas (KB_RETRIEVAL_PER_SOURCE_QUOTAS=True by default): runs one KNN
  scan per source bucket and caps each bucket at its configured quota. This prevents
  high-volume social chunks from crowding out papers/patents/uploads. Slack from
  underfilled buckets is NOT redistributed — simpler and avoids over-representing any
  single source when others have limited relevant content.
- Legacy global top_k path (KB_RETRIEVAL_PER_SOURCE_QUOTAS=False) is fully preserved
  for rollback — controlled by the feature flag in config.
"""

import asyncio
import json

import structlog

from app.core.cache_keys import stable_hash
from app.core.cache_version import get_project_cache_version
from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.redis import get_redis
from app.core.utils import parse_metadata
from app.kb.embedder import embed_texts
from app.repositories import knowledge as knowledge_repo

logger = structlog.get_logger(__name__)

# Source bucket definitions — each entry maps a logical bucket name to the list of
# source strings stored in knowledge_chunks.source for that bucket.
#
# IMPORTANT: these strings must exactly match the `source` column values written by
# the ingest pipeline.  Social sources are stored with the "social_" prefix
# (e.g. "social_tiktok", not "tiktok").  Web search results from Tavily/Exa are
# stored as "search" and are merged into the web_news bucket because they are
# semantically the same content type.
_SOURCE_BUCKETS: list[tuple[str, list[str]]] = [
    (
        "social",
        ["social_tiktok", "social_instagram", "social_x", "social_reddit", "social_youtube"],
    ),
    ("papers", ["paper"]),
    ("web_news", ["web", "news", "search"]),
    ("patents", ["patent"]),
    ("uploads", ["upload"]),
]


def _bucket_quota(bucket_name: str) -> int:
    """Return the configured per-bucket chunk quota."""
    return {
        "social": settings.KB_RETRIEVAL_QUOTA_SOCIAL,
        "papers": settings.KB_RETRIEVAL_QUOTA_PAPERS,
        "web_news": settings.KB_RETRIEVAL_QUOTA_WEB_NEWS,
        "patents": settings.KB_RETRIEVAL_QUOTA_PATENTS,
        "uploads": settings.KB_RETRIEVAL_QUOTA_UPLOADS,
    }.get(bucket_name, 0)


async def _retrieve_bucket(
    bucket_name: str,
    sources: list[str],
    project_ids: list[str],
    query_vector: str,
) -> list[dict]:
    """Fetch top-k chunks for one source bucket, applying per-bucket quota and threshold.

    Returns an empty list when the bucket quota is zero, no sources match, or all
    matching chunks score below KB_SCORE_THRESHOLD.
    """
    quota = _bucket_quota(bucket_name)
    if quota <= 0:
        return []

    rows = await knowledge_repo.retrieve_relevant_chunks_for_sources(
        project_ids=project_ids,
        query_vector=query_vector,
        top_k=quota,
        source_in=sources,
        prefer_fulltext=settings.ENABLE_FULLTEXT_RETRIEVAL_PREFERENCE,
    )

    results = []
    best_score: float | None = float(rows[0]["score"]) if rows else None
    for row in rows:
        score = float(row["score"])
        if score < settings.KB_SCORE_THRESHOLD:
            break  # rows are ordered by score DESC — stop at first below-threshold hit
        metadata = parse_metadata(row["metadata"])
        results.append(
            {
                "id": row["id"],
                "project_id": row["project_id"],
                "source": row["source"],
                "title": row["title"],
                "content": row["content"],
                "metadata": metadata,
                "score": round(score, 4),
            }
        )

    logger.debug(
        "retriever.bucket_result",
        bucket=bucket_name,
        count=len(results),
        best_score=round(best_score, 3) if best_score is not None else None,
        quota=quota,
    )
    return results


async def _retrieve_per_source(
    project_ids: list[str],
    query_vector: str,
    exclude_papers: bool,
) -> list[dict] | None:
    """Retrieve chunks using per-source bucket quotas.

    Runs all bucket scans concurrently. Returns None only when every bucket returns
    zero results — preserving the None == trigger-fallback contract of the global path.
    Logs per-bucket counts and best scores for observability.
    """
    active_buckets = [
        (name, sources)
        for name, sources in _SOURCE_BUCKETS
        if not (exclude_papers and name == "papers")
    ]

    # Fan out all bucket queries concurrently — one DB round-trip per bucket
    bucket_results: list[list[dict]] = await asyncio.gather(
        *[
            _retrieve_bucket(name, sources, project_ids, query_vector)
            for name, sources in active_buckets
        ]
    )

    combined: list[dict] = []
    bucket_counts: dict[str, int] = {}
    for (name, _), chunks in zip(active_buckets, bucket_results, strict=True):
        bucket_counts[name] = len(chunks)
        combined.extend(chunks)

    logger.info(
        "retriever.bucket_summary",
        bucket_counts=bucket_counts,
        total=len(combined),
        project_count=len(project_ids),
        exclude_papers=exclude_papers,
    )

    if not combined:
        return None

    return combined


async def retrieve(
    query: str,
    project_ids: list[str],
    top_k: int = settings.KB_RETRIEVAL_TOP_K,
    exclude_papers: bool = False,
) -> list[dict] | None:
    """
    Retrieve top-k most relevant chunks for a query across the given projects.

    Sets RLS app.accessible_projects so the policy is enforced even if the caller
    is the superuser. Returns None when the best score is below KB_SCORE_THRESHOLD —
    this signals the agent to trigger the search fallback instead of synthesising
    from weak context.

    When KB_RETRIEVAL_PER_SOURCE_QUOTAS is True (default), uses per-source bucket
    retrieval: one KNN scan per source group, each capped at its configured quota.
    This prevents high-volume social chunks from crowding out papers/patents/uploads.

    When KB_RETRIEVAL_PER_SOURCE_QUOTAS is False, falls back to the legacy global
    single-scan path with top_k chunks total (useful for rollback or testing).

    When exclude_papers=True, the papers bucket is skipped entirely in quota mode,
    or rows where source = 'paper' are excluded via SQL pre-filter in legacy mode.

    Each returned dict has: id, project_id, source, title, content, score, metadata.
    """
    if not query.strip():
        logger.warning("retriever.empty_query")
        return None

    if not project_ids:
        logger.warning("retriever.no_projects")
        return None

    # Check hot cache first — degrade gracefully if Redis is unreachable
    query_hash = stable_hash([query])
    hot_cache_key: str | None = None
    try:
        redis = await get_redis()
        sorted_project_ids = sorted(project_ids)
        project_key_part = stable_hash(sorted_project_ids)
        versions = []
        for pid in sorted_project_ids:
            versions.append(await get_project_cache_version(redis, pid))
        version_part = stable_hash(versions)[:8]
        # Cache variant encodes both the exclude_papers flag and the retrieval mode so
        # per-source and legacy results are never served interchangeably.
        quota_mode = "q" if settings.KB_RETRIEVAL_PER_SOURCE_QUOTAS else "g"
        cache_variant = f"{'xp' if exclude_papers else 'all'}:{quota_mode}"
        hot_cache_key = RedisKeys.KB_HOT.format(
            project_id=project_key_part, hash=f"{version_part}:{query_hash}:{cache_variant}"
        )
        cached = await redis.get(hot_cache_key)
        if cached:
            logger.info("retriever.hot_cache_hit", query_hash=query_hash)
            return json.loads(cached)
    except Exception:
        logger.warning("retriever.hot_cache_read_error", query_preview=query[:60])
        hot_cache_key = None  # skip cache write too

    # Embed the query — single text, uses embed cache
    vectors = await embed_texts([query])
    query_vector = vectors[0]

    # ── Per-source bucket path (default) ──────────────────────────────────────
    if settings.KB_RETRIEVAL_PER_SOURCE_QUOTAS:
        results = await _retrieve_per_source(
            project_ids=project_ids,
            query_vector=str(query_vector),
            exclude_papers=exclude_papers,
        )

        if results is None:
            logger.info("retriever.all_buckets_below_threshold", project_ids=project_ids)
            return None

        if hot_cache_key:
            try:
                redis = await get_redis()
                await redis.setex(hot_cache_key, RedisTTL.KB_HOT, json.dumps(results))
            except Exception:
                logger.warning("retriever.hot_cache_write_error", query_hash=query_hash)

        logger.info(
            "retriever.success",
            result_count=len(results),
            project_count=len(project_ids),
            exclude_papers=exclude_papers,
            mode="per_source_quotas",
        )
        return results

    # ── Legacy global single-scan path (KB_RETRIEVAL_PER_SOURCE_QUOTAS=False) ─
    rows = await knowledge_repo.retrieve_relevant_chunks(
        project_ids=project_ids,
        query_vector=str(query_vector),
        top_k=top_k,
        exclude_papers=exclude_papers,
        prefer_fulltext=settings.ENABLE_FULLTEXT_RETRIEVAL_PREFERENCE,
    )

    if not rows:
        logger.info("retriever.no_results", project_ids=project_ids)
        return None

    best_score = float(rows[0]["score"])
    results = []
    for row in rows:
        if float(row["score"]) < settings.KB_SCORE_THRESHOLD:
            continue
        metadata = parse_metadata(row["metadata"])
        results.append(
            {
                "id": row["id"],
                "project_id": row["project_id"],
                "source": row["source"],
                "title": row["title"],
                "content": row["content"],
                "metadata": metadata,
                "score": round(float(row["score"]), 4),
            }
        )

    if not results:
        logger.info(
            "retriever.below_threshold",
            best_score=round(best_score, 3),
            threshold=settings.KB_SCORE_THRESHOLD,
        )
        return None

    if hot_cache_key:
        try:
            redis = await get_redis()
            await redis.setex(hot_cache_key, RedisTTL.KB_HOT, json.dumps(results))
        except Exception:
            logger.warning("retriever.hot_cache_write_error", query_hash=query_hash)

    logger.info(
        "retriever.success",
        result_count=len(results),
        best_score=round(best_score, 3),
        project_count=len(project_ids),
        exclude_papers=exclude_papers,
        fulltext_preference=settings.ENABLE_FULLTEXT_RETRIEVAL_PREFERENCE,
        mode="global_top_k",
    )
    return results
