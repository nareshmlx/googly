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
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
    """
    Retrieve Discover feed rows scored by relevance, recency, and quality.

    Uses project intent embedding cosine similarity when available.
    Falls back to recency-only ranking when no intent embedding exists.
    Supports pagination for scalable feed rendering.
    """
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)", str(project_id)
        )
        rows = await conn.fetch(
            """
            WITH project_embedding AS (
                SELECT intent_embedding
                FROM projects
                WHERE id = $1::uuid
            ),
            project_flags AS (
                SELECT
                    tiktok_enabled,
                    instagram_enabled,
                    youtube_enabled,
                    reddit_enabled,
                    x_enabled,
                    papers_enabled,
                    patents_enabled,
                    perigon_enabled,
                    tavily_enabled,
                    exa_enabled
                FROM projects
                WHERE id = $1::uuid
            ),
            scored AS (
                SELECT
                    kc.id::text AS item_id,
                    kc.source,
                    kc.title,
                    kc.content AS summary,
                    kc.metadata,
                    kc.created_at,
                    CASE
                        WHEN pe.intent_embedding IS NOT NULL
                        THEN 1.0 - (kc.embedding <=> pe.intent_embedding)
                        ELSE 0.5
                    END AS relevance,
                    EXP(
                        -EXTRACT(EPOCH FROM (NOW() - kc.created_at)) / (30.0 * 86400)
                    ) AS recency,
                    CASE kc.source
                        WHEN 'paper' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'citation_count') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'citation_count')::int
                                            ELSE 0
                                        END
                                    )
                                ) / 10.0
                            )
                        WHEN 'patent' THEN
                            EXP(
                                -GREATEST(
                                    0,
                                    EXTRACT(YEAR FROM NOW()) - CASE
                                        WHEN (kc.metadata->>'year') ~ '^[0-9]{4}$'
                                        THEN (kc.metadata->>'year')::int
                                        WHEN (kc.metadata->>'date') ~ '^[0-9]{4}'
                                        THEN LEFT(kc.metadata->>'date', 4)::int
                                        ELSE 2022
                                    END
                                ) / 5.0
                            )
                        WHEN 'social_tiktok' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'likes') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'likes')::int
                                            ELSE 0
                                        END
                                    )
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'views') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'views')::int
                                            ELSE 0
                                        END
                                    )
                                ) / 20.0
                            )
                        WHEN 'social_instagram' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'likes') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'likes')::int
                                            ELSE 0
                                        END
                                    )
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'views') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'views')::int
                                            ELSE 0
                                        END
                                    )
                                ) / 20.0
                            )
                        WHEN 'social_youtube' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'likes') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'likes')::int
                                            ELSE 0
                                        END
                                    )
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'views') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'views')::int
                                            ELSE 0
                                        END
                                    )
                                ) / 20.0
                            )
                        WHEN 'social_reddit' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'score') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'score')::int
                                            ELSE 0
                                        END
                                    )
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'comments') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'comments')::int
                                            ELSE 0
                                        END
                                    )
                                ) / 20.0
                            )
                        WHEN 'social_x' THEN
                            LEAST(
                                1.0,
                                LOG(
                                    1
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'likes') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'likes')::int
                                            ELSE 0
                                        END
                                    )
                                    + GREATEST(
                                        0,
                                        CASE
                                            WHEN (kc.metadata->>'retweets') ~ '^-?[0-9]+$'
                                            THEN (kc.metadata->>'retweets')::int
                                            ELSE 0
                                        END
                                    )
                                ) / 20.0
                            )
                        WHEN 'search' THEN
                            CASE
                                WHEN (kc.metadata->>'score') ~ '^-?[0-9]+(\\.[0-9]+)?$'
                                THEN (kc.metadata->>'score')::float
                                ELSE 0.5
                            END
                        ELSE 0.5
                    END AS quality
                FROM knowledge_chunks AS kc
                CROSS JOIN project_embedding AS pe
                CROSS JOIN project_flags AS pf
                WHERE kc.project_id = $1::uuid
                  AND kc.source IN (
                      'social_tiktok',
                      'social_instagram',
                      'social_youtube',
                      'social_reddit',
                      'social_x',
                      'paper',
                      'patent',
                      'news',
                      'search'
                  )
                  AND (
                      (kc.source = 'social_tiktok' AND pf.tiktok_enabled)
                      OR (kc.source = 'social_instagram' AND pf.instagram_enabled)
                      OR (kc.source = 'social_youtube' AND pf.youtube_enabled)
                      OR (kc.source = 'social_reddit' AND pf.reddit_enabled)
                      OR (kc.source = 'social_x' AND pf.x_enabled)
                      OR (kc.source = 'paper' AND pf.papers_enabled)
                      OR (kc.source = 'patent' AND pf.patents_enabled)
                      OR (kc.source = 'news' AND pf.perigon_enabled)
                      OR (kc.source = 'search' AND (pf.tavily_enabled OR pf.exa_enabled))
                  )
            )
            SELECT
                item_id,
                source,
                title,
                summary,
                metadata,
                created_at,
                relevance,
                recency,
                quality,
                 CASE
                    WHEN pe_exists.has_embedding AND source = 'paper'
                    THEN (relevance * 0.65 + quality * 0.30 + recency * 0.05)
                    WHEN pe_exists.has_embedding AND source = 'patent'
                    THEN (relevance * 0.60 + quality * 0.30 + recency * 0.10)
                    WHEN pe_exists.has_embedding
                    THEN (relevance * 0.60 + quality * 0.15 + recency * 0.25)
                    WHEN source = 'paper'
                    THEN (quality * 0.85 + recency * 0.15)
                    WHEN source = 'patent'
                    THEN (quality * 0.80 + recency * 0.20)
                    WHEN source LIKE 'social_%'
                    THEN (quality * 0.40 + recency * 0.60)
                    ELSE (quality * 0.20 + recency * 0.80)
                END AS score
            FROM scored
            CROSS JOIN (SELECT EXISTS(SELECT 1 FROM project_embedding WHERE intent_embedding IS NOT NULL) AS has_embedding) AS pe_exists
            ORDER BY score DESC
            LIMIT $2 OFFSET $3
            """,
            project_id,
            limit,
            offset,
        )
    return [dict(r) for r in rows]


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


async def update_project_metadata(
    pool: asyncpg.Pool,
    project_id: str,
    metadata: dict,
) -> None:
    """Overwrite project metadata JSON and bump updated_at."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE projects
            SET metadata = $1::jsonb,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            json.dumps(metadata),
            project_id,
        )


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


async def fetch_upload_chunk_summaries(
    pool: asyncpg.Pool,
    project_id: str,
    upload_ids: list[str],
    *,
    limit: int = 40,
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
