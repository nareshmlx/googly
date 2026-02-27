"""Repository queries for knowledge_source_assets tracking table."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import asyncpg


async def upsert_source_asset(
    pool: asyncpg.Pool,
    *,
    project_id: str,
    user_id: str,
    source: str,
    source_id: str,
    title: str,
    source_url: str,
    resolved_url: str,
    canonical_url: str,
    source_fetcher: str,
) -> str | None:
    """Insert or touch an asset row and return asset_id."""
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO knowledge_source_assets (
                project_id, user_id, source, source_id, title,
                source_url, resolved_url, canonical_url,
                fetch_status, extract_status, attempt_count,
                metadata, created_at, updated_at
            ) VALUES (
                $1::uuid, $2, $3, $4, $5,
                $6, $7, $8,
                'pending', 'pending', 0,
                $9::jsonb, $10, $10
            )
            ON CONFLICT (project_id, source, source_id, canonical_url)
            DO UPDATE SET
                resolved_url = EXCLUDED.resolved_url,
                source_url = EXCLUDED.source_url,
                title = EXCLUDED.title,
                updated_at = EXCLUDED.updated_at
            RETURNING id::text
            """,
            project_id,
            user_id,
            source,
            source_id,
            title,
            source_url,
            resolved_url,
            canonical_url,
            json.dumps({"source_fetcher": source_fetcher}),
            now,
        )
    return str(row["id"]) if row else None


async def fetch_source_asset(pool: asyncpg.Pool, asset_id: str) -> dict | None:
    """Fetch one source asset row by id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, project_id::text, user_id, source, source_id, title,
                   source_url, resolved_url, canonical_url,
                   asset_type, mime_type, blob_path, checksum_sha256, byte_size,
                   fetch_status, extract_status, error_code, error_message,
                   attempt_count, next_attempt_at, last_attempt_at,
                   extracted_chars, extracted_pages, metadata
            FROM knowledge_source_assets
            WHERE id = $1::uuid
            """,
            asset_id,
        )
    return dict(row) if row else None


async def mark_fetch_result(
    pool: asyncpg.Pool,
    *,
    asset_id: str,
    fetch_status: str,
    mime_type: str,
    byte_size: int,
    checksum_sha256: str,
    blob_path: str,
    error_code: str = "",
    error_message: str = "",
) -> None:
    """Persist fetch lifecycle fields for one source asset."""
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE knowledge_source_assets
            SET fetch_status = $2,
                mime_type = $3,
                byte_size = $4,
                checksum_sha256 = $5,
                blob_path = $6,
                error_code = NULLIF($7, ''),
                error_message = NULLIF($8, ''),
                updated_at = $9
            WHERE id = $1::uuid
            """,
            asset_id,
            fetch_status,
            mime_type,
            int(byte_size),
            checksum_sha256,
            blob_path,
            error_code,
            error_message,
            now,
        )


async def mark_extract_result(
    pool: asyncpg.Pool,
    *,
    asset_id: str,
    extract_status: str,
    extracted_chars: int,
    extracted_pages: int,
    error_code: str = "",
    error_message: str = "",
) -> None:
    """Persist extraction lifecycle fields for one source asset."""
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE knowledge_source_assets
            SET extract_status = $2,
                extracted_chars = $3,
                extracted_pages = $4,
                error_code = CASE WHEN $2 = 'success' THEN NULL ELSE NULLIF($5, '') END,
                error_message = CASE WHEN $2 = 'success' THEN NULL ELSE NULLIF($6, '') END,
                updated_at = $7
            WHERE id = $1::uuid
            """,
            asset_id,
            extract_status,
            int(extracted_chars),
            int(extracted_pages),
            error_code,
            error_message,
            now,
        )


async def increment_attempt(
    pool: asyncpg.Pool,
    *,
    asset_id: str,
    next_attempt_at: datetime | None = None,
) -> int:
    """Increment attempt_count and return the new value."""
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE knowledge_source_assets
            SET attempt_count = attempt_count + 1,
                last_attempt_at = $2,
                next_attempt_at = $3,
                updated_at = $2
            WHERE id = $1::uuid
            RETURNING attempt_count
            """,
            asset_id,
            now,
            next_attempt_at,
        )
    return int((row or {}).get("attempt_count") or 0)


async def fetch_project_enrichment_counts(pool: asyncpg.Pool, project_id: str) -> dict:
    """Return lightweight fulltext enrichment funnel counts for one project."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE fetch_status IN ('pending', 'queued', 'retry'))::int AS pending,
                COUNT(*) FILTER (WHERE fetch_status = 'fetched')::int AS fetched,
                COUNT(*) FILTER (WHERE extract_status = 'success')::int AS extracted,
                COUNT(*) FILTER (
                    WHERE fetch_status = 'failed' OR extract_status IN ('failed', 'unsupported')
                )::int AS failed
            FROM knowledge_source_assets
            WHERE project_id = $1::uuid
            """,
            project_id,
        )
        embedded = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT metadata->>'asset_id')::int
            FROM knowledge_chunks
            WHERE project_id = $1::uuid
              AND source IN ('paper', 'patent')
              AND COALESCE(metadata->>'content_level', 'abstract') = 'fulltext'
              AND (metadata->>'asset_id') IS NOT NULL
            """,
            project_id,
        )
    payload = dict(row) if row else {"pending": 0, "fetched": 0, "extracted": 0, "failed": 0}
    payload["embedded"] = int(embedded or 0)
    return payload
