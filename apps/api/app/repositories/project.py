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

logger = structlog.get_logger(__name__)


async def insert_project(
    pool: asyncpg.Pool,
    user_id: str,
    title: str,
    description: str,
    refresh_strategy: str,
    tiktok_enabled: bool = True,
    instagram_enabled: bool = True,
    openalex_enabled: bool = True,
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
                 tiktok_enabled, instagram_enabled, openalex_enabled,
                 created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, 0, $7, $8, $9, $10, $10)
            RETURNING id::text, user_id, title, description,
                      refresh_strategy, structured_intent,
                      kb_chunk_count,
                      tiktok_enabled, instagram_enabled, openalex_enabled,
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
            openalex_enabled,
            now,
        )
    return dict(row)


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
                   kb_chunk_count, tiktok_enabled, instagram_enabled, openalex_enabled,
                   created_at, updated_at, last_refreshed_at
            FROM projects
            WHERE id = $1::uuid AND user_id = $2
            """,
            project_id,
            user_id,
        )
    return dict(row) if row else None


async def fetch_project_openalex_enabled(
    pool: asyncpg.Pool,
    project_id: str,
) -> bool | None:
    """
    Fetch openalex_enabled flag for one project.

    Returns None when project does not exist.
    """
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            """
            SELECT openalex_enabled
            FROM projects
            WHERE id = $1::uuid
            """,
            project_id,
        )
    return bool(value) if value is not None else None


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
                   kb_chunk_count, tiktok_enabled, instagram_enabled, openalex_enabled,
                   created_at, updated_at, last_refreshed_at
            FROM projects
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]


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
) -> list[dict]:
    """
    Fetch social KB chunks for the discover feed for one project.

    Sets app.accessible_projects before querying knowledge_chunks so the query
    follows the same RLS access policy as other KB reads.
    """
    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
        rows = await conn.fetch(
            """
            WITH base_social AS (
                SELECT source, source_id, title, content, metadata, created_at,
                    coalesce(nullif(metadata->>'username', ''), nullif(title, ''), source_id) AS creator_key,
                    CASE
                        WHEN (metadata->>'timestamp') ~ '^[0-9]+(\\.[0-9]+)?$'
                        THEN to_timestamp((metadata->>'timestamp')::double precision)
                        ELSE created_at
                    END AS content_ts,
                    ln(
                        1.0
                        + COALESCE((metadata->>'likes')::numeric, (metadata->>'like_count')::numeric, 0)
                        + COALESCE((metadata->>'views')::numeric, (metadata->>'view_count')::numeric, 0) / 10.0
                    )
                    * exp(
                        -EXTRACT(EPOCH FROM (now() - (
                            CASE
                                WHEN (metadata->>'timestamp') ~ '^[0-9]+(\\.[0-9]+)?$'
                                THEN to_timestamp((metadata->>'timestamp')::double precision)
                                ELSE created_at
                            END
                        ))) / (30.0 * 86400)
                    ) AS _score
                FROM knowledge_chunks
                WHERE project_id = $1::uuid
                  AND source IN ('social_tiktok', 'social_instagram')
            ),
            ig_ranked AS (
                SELECT source, source_id, title, content, metadata, created_at, creator_key, content_ts, _score,
                    row_number() OVER (PARTITION BY creator_key ORDER BY _score DESC) AS creator_rank
                FROM base_social
                WHERE source = 'social_instagram'
            ),
            ig_guaranteed AS (
                SELECT source, source_id, title, content, metadata, created_at, content_ts, _score
                FROM ig_ranked
                WHERE creator_rank <= 2
                ORDER BY _score DESC
                LIMIT 15
            ),
            top_rest_candidates AS (
                SELECT b.source, b.source_id, b.title, b.content, b.metadata, b.created_at, b.content_ts, b._score
                FROM base_social b
                LEFT JOIN ig_ranked ir ON ir.source = b.source AND ir.source_id = b.source_id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM ig_guaranteed g
                    WHERE g.source = b.source AND g.source_id = b.source_id
                )
                  AND (
                      b.source = 'social_tiktok'
                      OR (b.source = 'social_instagram' AND coalesce(ir.creator_rank, 999) <= 3)
                  )
            ),
            top_rest AS (
                SELECT source, source_id, title, content, metadata, created_at, content_ts, _score
                FROM top_rest_candidates
                ORDER BY _score DESC
                LIMIT 60
            )
            SELECT source, source_id, title, content, metadata, created_at
            FROM ig_guaranteed
            UNION ALL
            SELECT source, source_id, title, content, metadata, created_at
            FROM top_rest
            """,
            project_id,
        )
    return [dict(r) for r in rows]


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
