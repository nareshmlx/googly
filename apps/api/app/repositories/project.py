"""ProjectRepository — all asyncpg queries for the projects table.

Rule: This file contains ONLY SQL. No business logic. No intent extraction.
No ARQ calls. If a function needs more than one query, that logic belongs
in ProjectService, not here.

All functions accept an asyncpg Pool (or connection) as first argument so
they are easy to test with a real or mock pool.
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import asyncpg
import structlog

from app.core.config import settings
from app.core.db import get_db_pool

logger = structlog.get_logger(__name__)


async def _with_service_pool(fn, *args, **kwargs):
    """Execute a pool-injected repository function using the shared DB pool."""
    pool = await get_db_pool()
    return await fn(pool, *args, **kwargs)


async def insert_project(
    pool: asyncpg.Pool,
    user_id: str,
    title: str,
    description: str,
    refresh_strategy: str,
    tiktok_enabled: bool,
    instagram_enabled: bool,
    youtube_enabled: bool,
    reddit_enabled: bool,
    x_enabled: bool,
    papers_enabled: bool,
    patents_enabled: bool,
    perigon_enabled: bool,
    tavily_enabled: bool,
    exa_enabled: bool,
) -> dict:
    """
    Insert a new project row and return it as a dict.

    structured_intent starts as {} — caller updates it after extraction.
    Raises asyncpg.PostgresError on DB failure — caller must handle.
    """
    project_id = str(uuid4())
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO projects
                (id, user_id, title, description, refresh_strategy,
                 structured_intent, kb_chunk_count,
                 tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled,
                 papers_enabled, patents_enabled, perigon_enabled, tavily_enabled, exa_enabled,
                 metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, 0, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17::jsonb, $18, $18)
            RETURNING id::text, user_id, title, description,
                      refresh_strategy, structured_intent,
                      kb_chunk_count,
                      tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled, papers_enabled,
                      patents_enabled, perigon_enabled, tavily_enabled, exa_enabled,
                      metadata,
                      created_at, updated_at,
                      last_refreshed_at
            """,
            project_id,
            user_id,
            title,
            description,
            refresh_strategy,
            json.dumps({}),
            tiktok_enabled,
            instagram_enabled,
            youtube_enabled,
            reddit_enabled,
            x_enabled,
            papers_enabled,
            patents_enabled,
            perigon_enabled,
            tavily_enabled,
            exa_enabled,
            json.dumps({}),
            now,
        )
    return dict(row)


async def insert_project_for_service(
    user_id: str,
    title: str,
    description: str,
    refresh_strategy: str,
    tiktok_enabled: bool,
    instagram_enabled: bool,
    youtube_enabled: bool,
    reddit_enabled: bool,
    x_enabled: bool,
    papers_enabled: bool,
    patents_enabled: bool,
    perigon_enabled: bool,
    tavily_enabled: bool,
    exa_enabled: bool,
) -> dict:
    """Insert project with internally managed pool for service-layer calls."""
    return await _with_service_pool(
        insert_project,
        user_id=user_id,
        title=title,
        description=description,
        refresh_strategy=refresh_strategy,
        tiktok_enabled=tiktok_enabled,
        instagram_enabled=instagram_enabled,
        youtube_enabled=youtube_enabled,
        reddit_enabled=reddit_enabled,
        x_enabled=x_enabled,
        papers_enabled=papers_enabled,
        patents_enabled=patents_enabled,
        perigon_enabled=perigon_enabled,
        tavily_enabled=tavily_enabled,
        exa_enabled=exa_enabled,
    )


async def fetch_project(
    pool: asyncpg.Pool,
    project_id: str,
    user_id: str,
) -> dict | None:
    """
    Fetch a single project by ID, scoped to the owning user.

    Returns None if the project does not exist or belongs to a different user.
    Returning None (not raising) lets the service layer return a clean 404.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, user_id, title, description,
                   refresh_strategy, structured_intent,
                   kb_chunk_count, tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled, papers_enabled,
                   patents_enabled, perigon_enabled, tavily_enabled, exa_enabled,
                   metadata,
                   created_at, updated_at, last_refreshed_at
            FROM projects
            WHERE id = $1::uuid AND user_id = $2
            """,
            project_id,
            user_id,
        )
    return dict(row) if row else None


async def fetch_project_by_id(
    pool: asyncpg.Pool,
    project_id: str,
) -> dict | None:
    """Fetch a project by ID without user scoping (internal service use only)."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, user_id, title, description,
                   refresh_strategy, structured_intent,
                   kb_chunk_count, tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled, papers_enabled,
                   patents_enabled, perigon_enabled, tavily_enabled, exa_enabled,
                   metadata,
                   created_at, updated_at, last_refreshed_at
            FROM projects
            WHERE id = $1::uuid
            """,
            project_id,
        )
    return dict(row) if row else None


async def fetch_project_for_service(project_id: str, user_id: str) -> dict | None:
    """Fetch project for service-layer ownership checks with internal pool."""
    return await _with_service_pool(fetch_project, project_id, user_id)


async def fetch_project_by_id_for_service(project_id: str) -> dict | None:
    """Fetch project by ID with internally managed pool for service-layer calls."""
    return await _with_service_pool(fetch_project_by_id, project_id)


async def get_project_for_ingestion(
    pool: asyncpg.Pool,
    project_id: str,
) -> dict | None:
    """Fetch ingest-critical project fields for ingestion workers."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, user_id::text, title, description, structured_intent,
                   tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled,
                   papers_enabled, patents_enabled, perigon_enabled, tavily_enabled, exa_enabled
            FROM projects
            WHERE id = $1::uuid
            """,
            project_id,
        )
    return dict(row) if row else None


async def get_project_for_ingestion_for_service(project_id: str) -> dict | None:
    """Fetch ingestion project fields with internally managed pool."""
    return await _with_service_pool(get_project_for_ingestion, project_id)


async def project_exists(
    pool: asyncpg.Pool,
    project_id: str,
) -> bool:
    """Return True when the project exists."""
    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM projects WHERE id = $1::uuid", project_id)
    return bool(exists)


async def project_exists_for_service(project_id: str) -> bool:
    """Check project existence with internally managed pool."""
    return await _with_service_pool(project_exists, project_id)


async def get_chunk_count(
    pool: asyncpg.Pool,
    project_id: str,
) -> int:
    """Return total knowledge chunk count for a project."""
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*)::int FROM knowledge_chunks WHERE project_id = $1::uuid",
            project_id,
        )
    return int(count or 0)


async def get_chunk_count_for_service(project_id: str) -> int:
    """Return chunk count with internally managed pool."""
    return await _with_service_pool(get_chunk_count, project_id)


async def get_project_last_refreshed(
    pool: asyncpg.Pool,
    project_id: str,
) -> datetime | None:
    """Return project's last_refreshed_at timestamp if the project exists."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT last_refreshed_at FROM projects WHERE id = $1::uuid",
            project_id,
        )


async def get_project_last_refreshed_for_service(project_id: str) -> datetime | None:
    """Return last refresh timestamp with internally managed pool."""
    return await _with_service_pool(get_project_last_refreshed, project_id)


async def get_projects_due_for_refresh(pool: asyncpg.Pool) -> list[dict]:
    """Return projects due for daily/weekly refresh runs."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, last_refreshed_at FROM projects
            WHERE
                (refresh_strategy = 'daily'
                 AND (last_refreshed_at IS NULL OR last_refreshed_at < NOW() - INTERVAL '1 day'))
            OR
                (refresh_strategy = 'weekly'
                 AND (last_refreshed_at IS NULL OR last_refreshed_at < NOW() - INTERVAL '7 days'))
            """
        )
    return [dict(row) for row in rows]


async def get_projects_due_for_refresh_for_service() -> list[dict]:
    """List projects due for refresh with internally managed pool."""
    return await _with_service_pool(get_projects_due_for_refresh)


async def fetch_project_papers_enabled(
    pool: asyncpg.Pool,
    project_id: str,
) -> bool | None:
    """
    Fetch papers_enabled flag for one project.

    Returns None when project does not exist.
    """
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            """
            SELECT papers_enabled
            FROM projects
            WHERE id = $1::uuid
            """,
            project_id,
        )
    return bool(value) if value is not None else None


async def fetch_project_papers_enabled_for_chat(project_id: str) -> bool | None:
    """Repository convenience wrapper for chat service to avoid direct pool access."""
    return await _with_service_pool(fetch_project_papers_enabled, project_id)


async def list_projects(
    pool: asyncpg.Pool,
    user_id: str,
) -> list[dict]:
    """
    List all projects belonging to a user, ordered newest first.

    Returns an empty list (not None) when the user has no projects.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, user_id, title, description,
                   refresh_strategy, structured_intent,
                   kb_chunk_count, tiktok_enabled, instagram_enabled, youtube_enabled, reddit_enabled, x_enabled, papers_enabled,
                   patents_enabled, perigon_enabled, tavily_enabled, exa_enabled,
                   metadata,
                   created_at, updated_at, last_refreshed_at
            FROM projects
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]


async def list_projects_for_service(user_id: str) -> list[dict]:
    """List projects with internally managed pool for service-layer calls."""
    return await _with_service_pool(list_projects, user_id)


async def list_projects_summary_for_chat(user_id: str) -> list[dict]:
    """Repository convenience wrapper for chat service to avoid direct pool access."""
    return await _with_service_pool(list_projects_summary, user_id)


async def list_projects_summary(
    pool: asyncpg.Pool,
    user_id: str,
) -> list[dict]:
    """
    Return lightweight project summaries (id, title, description only).

    Used by ProjectRelevanceAgent — it only needs enough text to decide
    relevance, not full structured_intent or chunk counts.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, title, description
            FROM projects
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]


async def fetch_discover_feed(
    pool: asyncpg.Pool,
    project_id: str,
    limit: int = settings.PROJECT_DISCOVER_LIMIT,
    offset: int = 0,
) -> list[dict]:
    """
    Retrieve Discover feed rows scored by relevance, recency, and quality.

    Now queries 'knowledge_documents' directly for deduplication and performance.
    """
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)", str(project_id)
        )
        rows = await conn.fetch(
            """
            WITH project_embedding AS (
                SELECT intent_embedding FROM projects WHERE id = $1::uuid
            ),
            project_flags AS (
                SELECT
                    tiktok_enabled, instagram_enabled, youtube_enabled,
                    reddit_enabled, x_enabled, papers_enabled, patents_enabled,
                    perigon_enabled, exa_enabled, tavily_enabled
                FROM projects WHERE id = $1::uuid
            ),
                scored_documents AS (
                    SELECT DISTINCT ON (kd.id)
                    kd.id AS item_id,
                    kd.source,
                    kd.source_id,
                    kd.source_id AS base_source_id, -- Backward compatibility
                    COALESCE(kd.title, '') AS title,
                    COALESCE(kd.summary, '') AS summary,
                    kd.metadata,
                    kd.created_at,
                    -- Relevance: cosine similarity with project intent
                    CASE
                        WHEN pe.intent_embedding IS NOT NULL AND kc.embedding IS NOT NULL
                        THEN 1 - (kc.embedding <=> pe.intent_embedding)
                        ELSE 0.5
                    END AS relevance,
                    -- Recency: linear decay over 30 days
                    GREATEST(0, 1.0 - (EXTRACT(EPOCH FROM (NOW() - kd.created_at)) / (30 * 24 * 3600))) AS recency,
                    -- Quality: source-specific engagement normalization
                    CASE kd.source
                        WHEN 'paper' THEN
                            LEAST(
                                1.0,
                                (
                                    (
                                        CASE
                                            WHEN COALESCE(kd.metadata->>'citation_count', '') ~ '^-?[0-9]+$'
                                            THEN (kd.metadata->>'citation_count')::int
                                            ELSE 0
                                        END
                                    ) / 100.0
                                ) + (
                                    (
                                        CASE
                                            WHEN COALESCE(kd.metadata->>'influential_citation_count', '') ~ '^-?[0-9]+$'
                                            THEN (kd.metadata->>'influential_citation_count')::int
                                            ELSE 0
                                        END
                                    ) / 10.0
                                )
                            )
                        WHEN 'social_reddit' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1 + GREATEST(
                                        0,
                                        (
                                            CASE
                                                WHEN COALESCE(kd.metadata->>'score', '') ~ '^-?[0-9]+$'
                                                THEN (kd.metadata->>'score')::int
                                                ELSE 0
                                            END
                                        ) + (
                                            CASE
                                                WHEN COALESCE(kd.metadata->>'comments', '') ~ '^-?[0-9]+$'
                                                THEN (kd.metadata->>'comments')::int
                                                ELSE 0
                                            END
                                        )
                                    )
                                ) / 20.0
                            )
                        WHEN 'social_x' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1 + GREATEST(
                                        0,
                                        (
                                            CASE
                                                WHEN COALESCE(kd.metadata->>'likes', '') ~ '^-?[0-9]+$'
                                                THEN (kd.metadata->>'likes')::int
                                                ELSE 0
                                            END
                                        ) + (
                                            CASE
                                                WHEN COALESCE(kd.metadata->>'retweets', '') ~ '^-?[0-9]+$'
                                                THEN (kd.metadata->>'retweets')::int
                                                ELSE 0
                                            END
                                        )
                                    )
                                ) / 20.0
                            )
                        WHEN 'search' THEN
                            CASE
                                WHEN COALESCE(kd.metadata->>'score', '') ~ '^-?[0-9]*\\.?[0-9]+$'
                                THEN (kd.metadata->>'score')::double precision
                                ELSE 0.5
                            END
                        ELSE 0.5
                    END AS quality
                FROM knowledge_documents AS kd
                CROSS JOIN project_embedding AS pe
                CROSS JOIN project_flags AS pf
                LEFT JOIN LATERAL (
                    SELECT c.embedding, c.metadata
                    FROM knowledge_chunks AS c
                    WHERE c.document_id = kd.id
                      AND COALESCE(c.metadata->>'content_level', 'abstract') <> 'fulltext'
                    ORDER BY
                        CASE
                            WHEN pe.intent_embedding IS NOT NULL AND c.embedding IS NOT NULL
                            THEN c.embedding <=> pe.intent_embedding
                            ELSE NULL
                        END ASC NULLS LAST
                    LIMIT 1
                ) AS kc ON TRUE
                WHERE kd.project_id = $1::uuid
                  AND (
                      (kd.source = 'social_tiktok' AND pf.tiktok_enabled)
                      OR (kd.source = 'social_instagram' AND pf.instagram_enabled)
                      OR (kd.source = 'social_youtube' AND pf.youtube_enabled)
                      OR (kd.source = 'social_reddit' AND pf.reddit_enabled)
                      OR (kd.source = 'social_x' AND pf.x_enabled)
                      OR (kd.source = 'paper' AND pf.papers_enabled)
                      OR (kd.source = 'patent' AND pf.patents_enabled)
                      OR (kd.source = 'news' AND pf.perigon_enabled)
                      OR (kd.source = 'search' AND (pf.tavily_enabled OR pf.exa_enabled))
                  )
                ORDER BY kd.id, relevance DESC
            ),
            final_ranked AS (
                SELECT
                    *,
                    CASE
                        WHEN source = 'paper' THEN (relevance * 0.65 + quality * 0.30 + recency * 0.05)
                        WHEN source = 'patent' THEN (relevance * 0.60 + quality * 0.30 + recency * 0.10)
                        WHEN source LIKE 'social_%' THEN (relevance * 0.40 + quality * 0.40 + recency * 0.20)
                        ELSE (relevance * 0.50 + quality * 0.15 + recency * 0.35)
                    END AS score
                FROM scored_documents
            )
            SELECT
                item_id,
                source,
                source_id,
                base_source_id,
                title,
                summary,
                metadata,
                created_at,
                relevance,
                recency,
                quality,
                score
            FROM final_ranked
            ORDER BY score DESC
            LIMIT $2 OFFSET $3
            """,
            project_id,
            limit,
            offset,
        )
    return [dict(r) for r in rows]


async def fetch_discover_feed_for_service(
    project_id: str,
    limit: int = settings.PROJECT_DISCOVER_LIMIT,
    offset: int = 0,
) -> list[dict]:
    """Fetch discover feed with internally managed pool for service-layer calls."""
    return await _with_service_pool(fetch_discover_feed, project_id, limit=limit, offset=offset)


async def update_project_intent_embedding(
    pool: asyncpg.Pool,
    project_id: str,
    embedding: list[float],
) -> None:
    """Store intent embedding vector for project-level discover scoring."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE projects
            SET intent_embedding = $2::vector,
                updated_at = NOW()
            WHERE id = $1::uuid
            """,
            project_id,
            str(embedding),
        )


async def update_project_intent_embedding_for_service(
    project_id: str,
    embedding: list[float],
) -> None:
    """Store project intent embedding with internally managed pool for services."""
    await _with_service_pool(update_project_intent_embedding, project_id, embedding)


async def update_project_intent(
    pool: asyncpg.Pool,
    project_id: str,
    structured_intent: dict,
) -> None:
    """
    Overwrite structured_intent for a project.

    Called after extract_intent (creation) and refine_intent (document upload).
    updated_at is bumped so clients can detect stale cache.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE projects
            SET structured_intent = $1::jsonb,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            json.dumps(structured_intent),
            project_id,
        )


async def update_project_intent_for_service(
    project_id: str,
    structured_intent: dict,
) -> None:
    """Update structured intent with internally managed pool for services."""
    await _with_service_pool(update_project_intent, project_id, structured_intent)


async def update_project_kb_stats(
    pool: asyncpg.Pool,
    project_id: str,
    chunk_count: int,
    refreshed_at: datetime,
) -> None:
    """
    Update kb_chunk_count and last_refreshed_at after a successful ingest run.

    chunk_count is the NEW total (not a delta) — caller must query the current
    count first if they want to add to it, or the ARQ task can use a subquery.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE projects
            SET kb_chunk_count = $1,
                last_refreshed_at = $2,
                updated_at = NOW()
            WHERE id = $3::uuid
            """,
            chunk_count,
            refreshed_at,
            project_id,
        )


async def update_project_kb_stats_for_service(
    project_id: str,
    chunk_count: int,
    refreshed_at: datetime,
) -> None:
    """Update KB stats with internally managed pool."""
    await _with_service_pool(update_project_kb_stats, project_id, chunk_count, refreshed_at)


async def increment_project_kb_stats(
    pool: asyncpg.Pool,
    project_id: str,
    chunk_delta: int,
    refreshed_at: datetime,
) -> None:
    """Increment kb_chunk_count by delta and update last_refreshed_at."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE projects
            SET kb_chunk_count = GREATEST(0, kb_chunk_count + $1),
                last_refreshed_at = $2,
                updated_at = NOW()
            WHERE id = $3::uuid
            """,
            int(chunk_delta),
            refreshed_at,
            project_id,
        )


async def increment_project_kb_stats_for_service(
    project_id: str,
    chunk_delta: int,
    refreshed_at: datetime,
) -> None:
    """Increment KB stats with internally managed pool."""
    await _with_service_pool(increment_project_kb_stats, project_id, chunk_delta, refreshed_at)


async def delete_project(
    pool: asyncpg.Pool,
    project_id: str,
    user_id: str,
) -> bool:
    """
    Delete a project owned by the given user.

    knowledge_chunks are cascade-deleted by the FK constraint — no separate
    delete needed. Returns True if a row was deleted, False if not found
    or not owned by the user (so the service can return 404 cleanly).
    """
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM projects
            WHERE id = $1::uuid AND user_id = $2
            """,
            project_id,
            user_id,
        )
    # result is 'DELETE N' — N=0 means nothing matched.
    # Guard against None (asyncpg execute should always return a string, but be defensive).
    return (result or "DELETE 0").split()[-1] != "0"


async def delete_project_for_service(project_id: str, user_id: str) -> bool:
    """Delete project with internally managed pool for service-layer calls."""
    return await _with_service_pool(delete_project, project_id, user_id)


async def update_project_setup_status_metadata(
    pool: asyncpg.Pool,
    project_id: str,
    setup_status: dict,
) -> None:
    """Persist latest setup status under metadata.setup_status."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE projects
            SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('setup_status', $1::jsonb),
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            json.dumps(setup_status),
            project_id,
        )


async def update_project_setup_status_metadata_for_service(
    project_id: str,
    setup_status: dict,
) -> None:
    """Persist setup status with internally managed pool for service-layer calls."""
    await _with_service_pool(update_project_setup_status_metadata, project_id, setup_status)


async def fetch_upload_chunk_summaries(
    pool: asyncpg.Pool,
    project_id: str,
    upload_ids: list[str],
    *,
    limit: int = settings.PROJECT_UPLOAD_SUMMARIES_LIMIT,
) -> list[str]:
    """Return recent upload chunk snippets for provided upload IDs."""
    if not upload_ids:
        return []
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)", str(project_id)
        )
        rows = await conn.fetch(
            """
            SELECT content
            FROM knowledge_chunks
            WHERE project_id = $1::uuid
              AND source = 'upload'
              AND source_id = ANY($2::text[])
            ORDER BY created_at DESC
            LIMIT $3
            """,
            project_id,
            upload_ids,
            limit,
        )
    return [
        str(row.get("content") or "")[:300] for row in rows if str(row.get("content") or "").strip()
    ]


async def fetch_upload_chunk_summaries_for_service(
    project_id: str,
    upload_ids: list[str],
    *,
    limit: int = settings.PROJECT_UPLOAD_SUMMARIES_LIMIT,
) -> list[str]:
    """Return upload chunk summaries with internally managed pool."""
    return await _with_service_pool(
        fetch_upload_chunk_summaries, project_id, upload_ids, limit=limit
    )
