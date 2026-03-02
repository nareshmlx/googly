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
- top_k default 8 is a practical sweet spot: enough context for synthesis without
  overwhelming the LLM context window.
"""

import hashlib
import json

import structlog

from app.core.cache_version import get_project_cache_version
from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.db import get_db_pool
from app.core.redis import get_redis
from app.kb.embedder import embed_texts

logger = structlog.get_logger(__name__)


async def retrieve(
    query: str,
    project_ids: list[str],
    top_k: int = 8,
    exclude_papers: bool = False,
) -> list[dict] | None:
    """
    Retrieve top-k most relevant chunks for a query across the given projects.

    Sets RLS app.accessible_projects so the policy is enforced even if the caller
    is the superuser. Returns None when the best score is below KB_SCORE_THRESHOLD —
    this signals the agent to trigger the search fallback instead of synthesising
    from weak context.

    When exclude_papers=True, rows where source = 'paper' are excluded from the
    pgvector query via a SQL pre-filter, so they never enter the KNN candidate set.

    Each returned dict has: id, project_id, source, title, content, score, metadata.
    """
    if not query.strip():
        logger.warning("retriever.empty_query")
        return None

    if not project_ids:
        logger.warning("retriever.no_projects")
        return None

    # Check hot cache first — degrade gracefully if Redis is unreachable
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    hot_cache_key: str | None = None
    try:
        redis = await get_redis()
        project_key_part = hashlib.sha256(",".join(sorted(project_ids)).encode()).hexdigest()[:16]
        versions = []
        for pid in sorted(project_ids):
            versions.append(await get_project_cache_version(redis, pid))
        version_part = hashlib.sha256(",".join(versions).encode()).hexdigest()[:8]
        # Include exclude_papers flag in the cache key so filtered and unfiltered
        # results are stored in separate slots and never served to the wrong caller.
        cache_variant = "xp" if exclude_papers else "all"
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

    pool = await get_db_pool()
    async with pool.acquire() as conn, conn.transaction():
        # Set RLS session variable using set_config() with a parameterised call to
        # prevent SQL injection — an f-string here would let a crafted project_id
        # escape the string literal and modify the GUC value arbitrarily.
        accessible = ",".join(project_ids)
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)",
            accessible,
        )

        # Cosine similarity search pre-filtered by project_id
        # The <=> operator is cosine distance (lower = more similar), so 1 - distance = similarity
        # When exclude_papers=True an extra pre-filter clause is added so rows with
        # source = 'paper' are pruned before the KNN scan, not after.
        paper_filter = "AND source != 'paper'" if exclude_papers else ""
        preference_order = ""
        if settings.ENABLE_FULLTEXT_RETRIEVAL_PREFERENCE:
            preference_order = (
                "CASE WHEN COALESCE(metadata->>'content_level', 'abstract') = 'fulltext' "
                "THEN 0 ELSE 1 END,"
            )
        rows = await conn.fetch(
            f"""
            SELECT
                id::text,
                project_id::text,
                source,
                title,
                content,
                metadata,
                1 - (embedding <=> $1::vector) AS score
            FROM knowledge_chunks
            WHERE project_id = ANY($2::uuid[])
            {paper_filter}
            ORDER BY {preference_order} embedding <=> $1::vector
            LIMIT $3
            """,
            str(query_vector),
            project_ids,
            top_k,
        )

    if not rows:
        logger.info("retriever.no_results", project_ids=project_ids)
        return None

    # Filter to only chunks that meet the score threshold configured in settings.
    # We do this in Python (not SQL) so we can log the best score for observability.
    # Using settings.KB_SCORE_THRESHOLD as the single source of truth — the old
    # KbScoreThreshold.DEFAULT constant was a duplicate that could drift.
    best_score = float(rows[0]["score"])
    results = []
    for row in rows:
        if float(row["score"]) < settings.KB_SCORE_THRESHOLD:
            continue

        # Parse metadata — can be JSON string or dict from database
        metadata_raw = row["metadata"]
        if isinstance(metadata_raw, str):
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            metadata = {}

        results.append({
            "id": row["id"],
            "project_id": row["project_id"],
            "source": row["source"],
            "title": row["title"],
            "content": row["content"],
            "metadata": metadata,
            "score": round(float(row["score"]), 4),
        })


    if not results:
        logger.info(
            "retriever.below_threshold",
            best_score=round(best_score, 3),
            threshold=settings.KB_SCORE_THRESHOLD,
        )
        return None

    # Write to hot cache — only if we got useful results and Redis was reachable
    if hot_cache_key:
        try:
            redis = await get_redis()
            await redis.setex(hot_cache_key, RedisTTL.KB_HOT.value, json.dumps(results))
        except Exception:
            logger.warning("retriever.hot_cache_write_error", query_hash=query_hash)

    logger.info(
        "retriever.success",
        result_count=len(results),
        best_score=round(best_score, 3),
        project_count=len(project_ids),
        exclude_papers=exclude_papers,
        fulltext_preference=settings.ENABLE_FULLTEXT_RETRIEVAL_PREFERENCE,
    )
    return results
