"""Knowledge repository — asyncpg queries for knowledge_chunks task access."""

from __future__ import annotations

import json
from datetime import datetime

import asyncpg

from app.core.db import get_db_pool


async def get_chunks_for_document(
    pool: asyncpg.Pool,
    project_id: str,
    doc_source_id: str,
) -> list[dict]:
    """Fetch recent chunks for one uploaded document plus total project chunk count."""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
        rows = await conn.fetch(
            """
            WITH recent_chunks AS (
                SELECT content FROM knowledge_chunks
                WHERE project_id = $1::uuid AND source_id LIKE $2
                ORDER BY created_at DESC
                LIMIT 20
            ),
            chunk_count AS (
                SELECT COUNT(*) AS total FROM knowledge_chunks
                WHERE project_id = $1::uuid
            )
            SELECT
                recent_chunks.content,
                chunk_count.total
            FROM recent_chunks
            CROSS JOIN chunk_count
            """,
            project_id,
            f"{doc_source_id}%",
        )
    return [dict(row) for row in rows]


async def get_chunks_for_document_for_service(
    project_id: str,
    doc_source_id: str,
) -> list[dict]:
    """Fetch document chunks with internally managed pool."""
    pool = await get_db_pool()
    return await get_chunks_for_document(pool, project_id, doc_source_id)


async def get_chunks_for_fulltext_backfill(
    pool: asyncpg.Pool,
    *,
    cursor: tuple[datetime, str],
    batch_size: int,
    project_id: str | None = None,
) -> list[dict]:
    """Fetch paginated paper/patent metadata chunks for fulltext backfill."""
    cursor_ts, cursor_id = cursor
    bounded_batch = max(1, min(int(batch_size), 500))
    async with pool.acquire() as conn:
        if project_id:
            rows = await conn.fetch(
                """
                SELECT kc.id::text, kc.project_id::text, kc.user_id, kc.source, kc.source_id,
                       kc.title, kc.metadata, kc.created_at
                FROM knowledge_chunks kc
                WHERE kc.source IN ('paper', 'patent')
                  AND COALESCE(kc.metadata->>'content_level', 'abstract') != 'fulltext'
                  AND (
                        kc.created_at > $1::timestamptz
                        OR (kc.created_at = $1::timestamptz AND kc.id::text > $2)
                  )
                  AND kc.project_id = $3::uuid
                ORDER BY kc.created_at ASC, kc.id::text ASC
                LIMIT $4
                """,
                cursor_ts,
                cursor_id,
                project_id,
                bounded_batch,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT kc.id::text, kc.project_id::text, kc.user_id, kc.source, kc.source_id,
                       kc.title, kc.metadata, kc.created_at
                FROM knowledge_chunks kc
                WHERE kc.source IN ('paper', 'patent')
                  AND COALESCE(kc.metadata->>'content_level', 'abstract') != 'fulltext'
                  AND (
                        kc.created_at > $1::timestamptz
                        OR (kc.created_at = $1::timestamptz AND kc.id::text > $2)
                  )
                ORDER BY kc.created_at ASC, kc.id::text ASC
                LIMIT $3
                """,
                cursor_ts,
                cursor_id,
                bounded_batch,
            )
    return [dict(row) for row in rows]


async def get_chunks_for_fulltext_backfill_for_service(
    *,
    cursor: tuple[datetime, str],
    batch_size: int,
    project_id: str | None = None,
) -> list[dict]:
    """Fetch backfill rows with internally managed pool."""
    pool = await get_db_pool()
    return await get_chunks_for_fulltext_backfill(
        pool,
        cursor=cursor,
        batch_size=batch_size,
        project_id=project_id,
    )


async def retrieve_relevant_chunks(
    *,
    project_ids: list[str],
    query_vector: str,
    top_k: int,
    exclude_papers: bool,
    prefer_fulltext: bool,
) -> list[dict]:
    """Run RLS-scoped pgvector retrieval query across project IDs."""
    pool = await get_db_pool()
    async with pool.acquire() as conn, conn.transaction():
        accessible = ",".join(project_ids)
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)",
            accessible,
        )

        paper_filter = "AND source != 'paper'" if exclude_papers else ""
        preference_order = ""
        if prefer_fulltext:
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
            query_vector,
            project_ids,
            top_k,
        )
    return [dict(row) for row in rows]


async def retrieve_relevant_chunks_for_sources(
    *,
    project_ids: list[str],
    query_vector: str,
    top_k: int,
    source_in: list[str],
    prefer_fulltext: bool,
) -> list[dict]:
    """Run RLS-scoped pgvector retrieval filtered to a specific set of source values.

    Used by the per-source quota retrieval strategy in the KB retriever.  Each bucket
    calls this function with its own source list and per-bucket top_k cap.  RLS is set
    per-call to maintain multi-tenant isolation.
    """
    if not source_in:
        return []

    pool = await get_db_pool()
    async with pool.acquire() as conn, conn.transaction():
        accessible = ",".join(project_ids)
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)",
            accessible,
        )

        preference_order = ""
        if prefer_fulltext:
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
              AND source = ANY($3::text[])
            ORDER BY {preference_order} embedding <=> $1::vector
            LIMIT $4
            """,
            query_vector,
            project_ids,
            source_in,
            top_k,
        )
    return [dict(row) for row in rows]


async def upsert_documents(records: list[dict]) -> dict[tuple[str, str, str | None], str]:
    """Bulk upsert knowledge_documents and return source key -> document id map."""
    if not records:
        return {}

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            INSERT INTO knowledge_documents (id, project_id, user_id, source, source_id, title, summary, metadata, created_at, updated_at)
            SELECT
                uuid_generate_v4(),
                unnest($1::uuid[]),
                unnest($2::text[]),
                unnest($3::text[]),
                unnest($4::text[]),
                unnest($5::text[]),
                unnest($7::text[]),
                unnest($6::text[])::jsonb,
                NOW(),
                NOW()
            ON CONFLICT (project_id, source, source_id) DO UPDATE SET
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING project_id::text, source, source_id, id::text
            """,
            [r["project_id"] for r in records],
            [r["user_id"] for r in records],
            [r["source"] for r in records],
            [r["source_id"] for r in records],
            [r["title"] for r in records],
            [json.dumps(r["metadata"]) for r in records],
            [r["summary"] for r in records],
        )
    return {
        (row["project_id"], row["source"], row["source_id"]): row["id"]
        for row in [dict(r) for r in rows]
    }


async def insert_chunks(chunks: list[dict], vectors: list[list[float]]) -> int:
    """Bulk insert knowledge chunks and return inserted row count."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        msg = "chunks and vectors must have the same length"
        raise ValueError(msg)
    if any(not c.get("document_id") for c in chunks):
        msg = "document_id is required for every chunk insert"
        raise ValueError(msg)

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        inserted = await conn.fetchval(
            """
            WITH inserted AS (
                INSERT INTO knowledge_chunks
                    (id, document_id, project_id, user_id, source, source_id, title,
                     content, embedding, metadata)
                SELECT
                    uuid_generate_v4(),
                    unnest($1::uuid[]),
                    unnest($2::uuid[]),
                    unnest($3::text[]),
                    unnest($4::text[]),
                    unnest($5::text[]),
                    unnest($6::text[]),
                    unnest($7::text[]),
                    unnest($8::text[])::vector,
                    unnest($9::text[])::jsonb
                ON CONFLICT (project_id, source, source_id) DO NOTHING
                RETURNING 1
            )
            SELECT COUNT(*)::int FROM inserted
            """,
            [c["document_id"] for c in chunks],
            [c["project_id"] for c in chunks],
            [c["user_id"] for c in chunks],
            [c["source"] for c in chunks],
            [c["source_id"] for c in chunks],
            [c["title"] for c in chunks],
            [c["content"] for c in chunks],
            [str(vectors[i]) for i in range(len(chunks))],
            [json.dumps(c["metadata"]) for c in chunks],
        )
    return int(inserted or 0)
