"""Insights repository — asyncpg queries for project_insights and related source context."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, TypeVar

import asyncpg

from app.core.db import get_db_pool

T = TypeVar("T")


def _as_uuid_list(values: list[str]) -> list[str]:
    """Return a sanitized UUID string list for SQL ANY/unnest usage."""
    return [str(v).strip() for v in values if str(v).strip()]


def _project_scope_value(project_ids: list[str]) -> str:
    """Build set_config value for app.accessible_projects."""
    return ",".join(_as_uuid_list(project_ids))


def _to_int_or_none(value: Any) -> int | None:
    """Convert loose numeric payload to int when possible."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    compact = (
        text.replace(",", "")
        .replace("views", "")
        .replace("view", "")
        .replace("likes", "")
        .replace("like", "")
        .strip()
    )
    multiplier = 1.0
    if compact.endswith("k"):
        compact = compact[:-1].strip()
        multiplier = 1_000.0
    elif compact.endswith("m"):
        compact = compact[:-1].strip()
        multiplier = 1_000_000.0
    elif compact.endswith("b"):
        compact = compact[:-1].strip()
        multiplier = 1_000_000_000.0
    try:
        return int(float(compact) * multiplier)
    except Exception:
        return None


async def _with_service_pool(fn, *args, **kwargs) -> T:  # type: ignore[no-untyped-def]
    """Run a connection-bound repository function with an internally managed pool."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await fn(conn, *args, **kwargs)  # type: ignore[no-any-return]


async def get_insights_for_project(conn: asyncpg.Connection, project_id: str) -> list[dict]:
    """Return all insight cards for one project ordered by cluster size descending."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    rows = await conn.fetch(
        """
        SELECT
            pi.id::text,
            pi.topic_label,
            pi.executive_summary,
            pi.key_findings,
            pi.trend_signal,
            pi.cluster_size,
            COALESCE(jsonb_array_length(pi.source_doc_ids), 0) AS source_doc_count,
            pi.full_report_status,
            COALESCE(doc_source_counts.source_type_counts, '{}'::jsonb) AS source_type_counts
        FROM project_insights pi
        LEFT JOIN LATERAL (
            SELECT jsonb_object_agg(src.source, src.doc_count) AS source_type_counts
            FROM (
                SELECT kd.source, COUNT(*)::int AS doc_count
                FROM jsonb_array_elements_text(COALESCE(pi.source_doc_ids, '[]'::jsonb)) sd(doc_id_text)
                JOIN knowledge_documents kd ON kd.id = sd.doc_id_text::uuid
                WHERE kd.project_id = pi.project_id
                GROUP BY kd.source
            ) src
        ) doc_source_counts ON TRUE
        WHERE pi.project_id = $1::uuid
        ORDER BY pi.cluster_size DESC, pi.created_at DESC
        """,
        project_id,
    )
    return [dict(row) for row in rows]


async def get_insights_for_project_for_service(project_id: str) -> list[dict]:
    """Service wrapper for get_insights_for_project."""
    return await _with_service_pool(get_insights_for_project, project_id)


async def get_insight_by_id(
    conn: asyncpg.Connection,
    project_id: str,
    insight_id: str,
) -> dict | None:
    """Return one insight row with deduplicated source document summaries."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    insight_row = await conn.fetchrow(
        """
        SELECT
            pi.id::text,
            pi.project_id::text,
            pi.topic_label,
            pi.executive_summary,
            pi.key_findings,
            pi.trend_signal,
            pi.contradictions,
            pi.chunk_ids,
            pi.source_doc_ids,
            pi.cluster_size,
            COALESCE(jsonb_array_length(pi.source_doc_ids), 0) AS source_doc_count,
            pi.full_report,
            pi.full_report_status,
            COALESCE(doc_source_counts.source_type_counts, '{}'::jsonb) AS source_type_counts
        FROM project_insights pi
        LEFT JOIN LATERAL (
            SELECT jsonb_object_agg(src.source, src.doc_count) AS source_type_counts
            FROM (
                SELECT kd.source, COUNT(*)::int AS doc_count
                FROM jsonb_array_elements_text(COALESCE(pi.source_doc_ids, '[]'::jsonb)) sd(doc_id_text)
                JOIN knowledge_documents kd ON kd.id = sd.doc_id_text::uuid
                WHERE kd.project_id = pi.project_id
                GROUP BY kd.source
            ) src
        ) doc_source_counts ON TRUE
        WHERE pi.project_id = $1::uuid
          AND pi.id = $2::uuid
        LIMIT 1
        """,
        project_id,
        insight_id,
    )
    if insight_row is None:
        return None

    row = dict(insight_row)
    doc_rows = await conn.fetch(
        """
        WITH source_docs AS (
            SELECT DISTINCT kd.id::text AS id
            FROM project_insights pi
            JOIN LATERAL jsonb_array_elements_text(pi.source_doc_ids) src(doc_id_text) ON TRUE
            JOIN knowledge_documents kd ON kd.id = src.doc_id_text::uuid
            WHERE pi.project_id = $1::uuid
              AND pi.id = $2::uuid
        )
        SELECT
            kd.id::text,
            kd.title,
            kd.source,
            NULLIF(kd.summary, '') AS stored_summary,
            COALESCE(
                NULLIF(kd.metadata->>'summary', ''),
                NULLIF(kd.metadata->>'snippet', ''),
                NULLIF(kc.metadata->>'summary', ''),
                NULLIF(kc.metadata->>'snippet', ''),
                NULLIF(LEFT(regexp_replace(COALESCE(kc.content, ''), E'\\s+', ' ', 'g'), 1200), '')
            ) AS summary_seed,
            COALESCE(
                kd.metadata->>'url',
                kd.metadata->>'link',
                kd.metadata->>'permalink',
                kd.metadata->>'video_url',
                kc.metadata->>'url',
                kc.metadata->>'link',
                kc.metadata->>'permalink',
                kc.metadata->>'video_url',
                ''
            ) AS url,
            COALESCE(
                kd.metadata->>'cover_url',
                kd.metadata->>'thumbnail_url',
                kd.metadata->>'image_url',
                kd.metadata->>'imageUrl',
                kd.metadata->>'preview_image_url',
                kd.metadata->>'thumbnail_src',
                kd.metadata->>'thumbnailUrl',
                kd.metadata->>'display_url',
                kd.metadata->>'thumbnail',
                kd.metadata->>'image',
                kc.metadata->>'cover_url',
                kc.metadata->>'thumbnail_url',
                kc.metadata->>'image_url',
                kc.metadata->>'imageUrl',
                kc.metadata->>'preview_image_url',
                kc.metadata->>'thumbnail_src',
                kc.metadata->>'thumbnailUrl',
                kc.metadata->>'display_url',
                kc.metadata->>'thumbnail',
                kc.metadata->>'image',
                ''
            ) AS cover_url,
            COALESCE(
                kd.metadata->>'video_url',
                kd.metadata->>'video',
                kd.metadata->>'videoUrl',
                kd.metadata->>'play_url',
                kc.metadata->>'video_url',
                kc.metadata->>'video',
                kc.metadata->>'videoUrl',
                kc.metadata->>'play_url',
                ''
            ) AS video_url,
            COALESCE(
                kd.metadata->>'author',
                kd.metadata->>'username',
                kd.metadata->>'channel_title',
                kd.metadata->>'channel',
                kc.metadata->>'author',
                kc.metadata->>'username',
                kc.metadata->>'channel_title',
                kc.metadata->>'channel',
                ''
            ) AS author,
            COALESCE(
                kd.metadata->>'views',
                kd.metadata->>'view_count',
                kd.metadata->>'play_count',
                kc.metadata->>'views',
                kc.metadata->>'view_count',
                kc.metadata->>'play_count',
                ''
            ) AS views,
            COALESCE(
                kd.metadata->>'likes',
                kd.metadata->>'like_count',
                kc.metadata->>'likes',
                kc.metadata->>'like_count',
                ''
            ) AS likes,
            COALESCE(
                kd.metadata->>'published_at',
                kd.metadata->>'published',
                kd.metadata->>'created_at',
                kd.metadata->>'create_time',
                kd.metadata->>'timestamp',
                kd.metadata->>'date',
                kc.metadata->>'published_at',
                kc.metadata->>'published',
                kc.metadata->>'created_at',
                kc.metadata->>'create_time',
                kc.metadata->>'timestamp',
                kc.metadata->>'date',
                ''
            ) AS published_at
        FROM source_docs sd
        JOIN knowledge_documents kd ON kd.id::text = sd.id
        LEFT JOIN LATERAL (
            SELECT kc.content, kc.metadata
            FROM knowledge_chunks kc
            WHERE kc.project_id = $1::uuid
              AND kc.document_id = kd.id
            ORDER BY kc.created_at DESC
            LIMIT 1
        ) kc ON TRUE
        ORDER BY kd.created_at DESC
        """,
        project_id,
        insight_id,
    )

    row["source_docs"] = [
        {
            "id": d["id"],
            "title": d["title"] or "",
            "source": d["source"] or "",
            "summary": d["stored_summary"] or d["summary_seed"],
            "summary_origin": (
                "generated"
                if d["stored_summary"]
                else ("fallback" if d["summary_seed"] else "missing")
            ),
            "url": d["url"] or None,
            "cover_url": d["cover_url"] or None,
            "video_url": d["video_url"] or None,
            "author": d["author"] or None,
            "views": _to_int_or_none(d["views"]),
            "likes": _to_int_or_none(d["likes"]),
            "published_at": d["published_at"] or None,
        }
        for d in doc_rows
    ]
    return row


async def get_insight_by_id_for_service(project_id: str, insight_id: str) -> dict | None:
    """Service wrapper for get_insight_by_id."""
    return await _with_service_pool(get_insight_by_id, project_id, insight_id)


async def get_insight_by_id_in_projects(
    conn: asyncpg.Connection,
    project_ids: list[str],
    insight_id: str,
) -> dict | None:
    """Return one insight row scoped to a provided project-id set."""
    scoped_ids = _as_uuid_list(project_ids)
    if not scoped_ids:
        return None
    await conn.execute(
        "SELECT set_config('app.accessible_projects', $1, true)",
        _project_scope_value(scoped_ids),
    )
    row = await conn.fetchrow(
        """
        SELECT
            pi.id::text,
            pi.project_id::text,
            pi.topic_label,
            pi.executive_summary,
            pi.key_findings,
            pi.trend_signal,
            pi.contradictions,
            pi.chunk_ids,
            pi.source_doc_ids,
            pi.cluster_size,
            pi.full_report,
            pi.full_report_status,
            COALESCE(doc_source_counts.source_type_counts, '{}'::jsonb) AS source_type_counts
        FROM project_insights pi
        LEFT JOIN LATERAL (
            SELECT jsonb_object_agg(src.source, src.doc_count) AS source_type_counts
            FROM (
                SELECT kd.source, COUNT(*)::int AS doc_count
                FROM jsonb_array_elements_text(COALESCE(pi.source_doc_ids, '[]'::jsonb)) sd(doc_id_text)
                JOIN knowledge_documents kd ON kd.id = sd.doc_id_text::uuid
                WHERE kd.project_id = pi.project_id
                GROUP BY kd.source
            ) src
        ) doc_source_counts ON TRUE
        WHERE pi.id = $1::uuid
          AND pi.project_id = ANY($2::uuid[])
        LIMIT 1
        """,
        insight_id,
        scoped_ids,
    )
    return dict(row) if row is not None else None


async def get_insight_by_id_in_projects_for_service(
    project_ids: list[str],
    insight_id: str,
) -> dict | None:
    """Service wrapper for get_insight_by_id_in_projects."""
    return await _with_service_pool(get_insight_by_id_in_projects, project_ids, insight_id)


async def get_chunks_for_insight(
    conn: asyncpg.Connection,
    project_id: str,
    chunk_ids: list[str],
) -> list[dict]:
    """Return chunk context rows for a specific insight chunk-id set."""
    ids = _as_uuid_list(chunk_ids)
    if not ids:
        return []
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    rows = await conn.fetch(
        """
        SELECT
            id::text,
            project_id::text,
            document_id::text,
            source,
            title,
            content,
            metadata,
            embedding::text AS embedding
        FROM knowledge_chunks
        WHERE project_id = $1::uuid
          AND id = ANY($2::uuid[])
        ORDER BY document_id ASC,
                 COALESCE((metadata->>'page_start')::int, 0) ASC,
                 COALESCE((metadata->>'published_at'), ''),
                 created_at ASC

        """,
        project_id,
        ids,
    )
    return [dict(row) for row in rows]


async def get_chunks_for_insight_for_service(project_id: str, chunk_ids: list[str]) -> list[dict]:
    """Service wrapper for get_chunks_for_insight."""
    return await _with_service_pool(get_chunks_for_insight, project_id, chunk_ids)


async def get_chunks_for_documents(
    conn: asyncpg.Connection,
    project_id: str,
    source_doc_ids: list[str],
    *,
    limit: int = 200,
) -> list[dict]:
    """Return all chunks for one insight's source-doc expansion scope."""
    doc_ids = _as_uuid_list(source_doc_ids)
    if not doc_ids:
        return []
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    rows = await conn.fetch(
        """
        SELECT
            id::text,
            project_id::text,
            document_id::text,
            source,
            title,
            content,
            metadata,
            embedding::text AS embedding
        FROM knowledge_chunks
        WHERE project_id = $1::uuid
          AND document_id = ANY($2::uuid[])
        ORDER BY created_at DESC
        LIMIT $3
        """,
        project_id,
        doc_ids,
        max(1, min(limit, 2000)),
    )
    return [dict(row) for row in rows]


async def get_chunks_for_documents_for_service(
    project_id: str,
    source_doc_ids: list[str],
    *,
    limit: int = 200,
) -> list[dict]:
    """Service wrapper for get_chunks_for_documents."""
    return await _with_service_pool(
        get_chunks_for_documents,
        project_id,
        source_doc_ids,
        limit=limit,
    )


async def get_cluster_candidate_chunks(
    conn: asyncpg.Connection,
    project_id: str,
    *,
    limit: int,
    max_per_document: int = 8,
) -> list[dict]:
    """Return latest embedding-backed chunks for project clustering."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    rows = await conn.fetch(
        """
        WITH ranked AS (
            SELECT
                id::text AS id,
                document_id::text AS document_id,
                source,
                title,
                content,
                metadata,
                embedding::text AS embedding,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY document_id
                    ORDER BY created_at DESC
                ) AS doc_rank
            FROM knowledge_chunks
            WHERE project_id = $1::uuid
              AND embedding IS NOT NULL
        )
        SELECT
            id,
            document_id,
            source,
            title,
            content,
            metadata,
            embedding
        FROM ranked
        WHERE doc_rank <= $3
        ORDER BY created_at DESC
        LIMIT $2
        """,
        project_id,
        limit,
        max(1, min(max_per_document, 100)),
    )
    return [dict(row) for row in rows]


async def get_cluster_candidate_chunks_for_service(
    project_id: str,
    *,
    limit: int,
    max_per_document: int = 8,
) -> list[dict]:
    """Service wrapper for get_cluster_candidate_chunks."""
    return await _with_service_pool(
        get_cluster_candidate_chunks,
        project_id,
        limit=limit,
        max_per_document=max_per_document,
    )


async def bulk_replace_insights(
    conn: asyncpg.Connection,
    project_id: str,
    rows: list[dict[str, Any]],
) -> None:
    """Atomically replace all insights for a project with a new clustered snapshot."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    async with conn.transaction():
        await conn.execute("DELETE FROM project_insights WHERE project_id = $1::uuid", project_id)
        if not rows:
            return

        now = datetime.now(UTC)
        records = [
            (
                row["id"],
                project_id,
                row["topic_label"],
                row["executive_summary"],
                json.dumps(row.get("key_findings", [])),
                row.get("trend_signal", "unknown"),
                row.get("contradictions"),
                json.dumps(row.get("chunk_ids", [])),
                json.dumps(row.get("source_doc_ids", [])),
                int(row.get("cluster_size") or 0),
                row.get("full_report"),
                row.get("full_report_status", "pending"),
                json.dumps(row.get("source_type_counts", {})),
                now,
                now,
            )
            for row in rows
        ]
        await conn.copy_records_to_table(
            "project_insights",
            records=records,
            columns=[
                "id",
                "project_id",
                "topic_label",
                "executive_summary",
                "key_findings",
                "trend_signal",
                "contradictions",
                "chunk_ids",
                "source_doc_ids",
                "cluster_size",
                "full_report",
                "full_report_status",
                "source_type_counts",
                "created_at",
                "updated_at",
            ],
        )


async def bulk_replace_insights_for_service(project_id: str, rows: list[dict[str, Any]]) -> None:
    """Service wrapper for bulk_replace_insights."""
    await _with_service_pool(bulk_replace_insights, project_id, rows)


async def update_full_report(
    conn: asyncpg.Connection,
    project_id: str,
    insight_id: str,
    full_report: str,
    status: str,
) -> None:
    """Update full report body and status for one insight."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    await conn.execute(
        """
        UPDATE project_insights
        SET full_report = $2,
            full_report_status = $3,
            updated_at = NOW()
        WHERE id = $1::uuid
          AND project_id = $4::uuid
        """,
        insight_id,
        full_report,
        status,
        project_id,
    )


async def update_full_report_for_service(
    project_id: str,
    insight_id: str,
    full_report: str,
    status: str,
) -> None:
    """Service wrapper for update_full_report."""
    await _with_service_pool(update_full_report, project_id, insight_id, full_report, status)


async def set_full_report_status(
    conn: asyncpg.Connection,
    project_id: str,
    insight_id: str,
    status: str,
) -> None:
    """Update only the full_report_status for one insight row."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    await conn.execute(
        """
        UPDATE project_insights
        SET full_report_status = $2,
            updated_at = NOW()
        WHERE id = $1::uuid
          AND project_id = $3::uuid
        """,
        insight_id,
        status,
        project_id,
    )


async def set_full_report_status_for_service(project_id: str, insight_id: str, status: str) -> None:
    """Service wrapper for set_full_report_status."""
    await _with_service_pool(set_full_report_status, project_id, insight_id, status)


async def claim_full_report_generation(
    conn: asyncpg.Connection,
    project_id: str,
    insight_id: str,
    max_reports_per_day: int | None = None,
) -> bool:
    """Atomically claim report generation ownership for a pending/failed insight."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    updated = await conn.fetchval(
        """
        WITH reports_generated_today AS (
            SELECT COUNT(*)::int AS report_count
            FROM project_insights
            WHERE project_id = $2::uuid
              AND full_report_status = 'done'
              AND updated_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
        )
        UPDATE project_insights
        SET full_report_status = 'generating',
            updated_at = NOW()
        WHERE id = $1::uuid
          AND project_id = $2::uuid
          AND full_report_status IN ('pending', 'failed')
          AND (
              $3::int IS NULL
              OR (SELECT report_count FROM reports_generated_today) < $3::int
          )
        RETURNING 1
        """,
        insight_id,
        project_id,
        max_reports_per_day,
    )
    return bool(updated)


async def claim_full_report_generation_for_service(
    project_id: str,
    insight_id: str,
    max_reports_per_day: int | None = None,
) -> bool:
    """Service wrapper for claim_full_report_generation."""
    return await _with_service_pool(
        claim_full_report_generation,
        project_id,
        insight_id,
        max_reports_per_day,
    )


async def count_reports_generated_today(conn: asyncpg.Connection, project_id: str) -> int:
    """Count completed full reports generated today (UTC) for one project."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    value = await conn.fetchval(
        """
        SELECT COUNT(*)::int
        FROM project_insights
        WHERE project_id = $1::uuid
          AND full_report_status = 'done'
          AND updated_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
        """,
        project_id,
    )
    return int(value or 0)


async def count_reports_generated_today_for_service(project_id: str) -> int:
    """Service wrapper for count_reports_generated_today."""
    return await _with_service_pool(count_reports_generated_today, project_id)


async def list_projects_needing_insight_backfill(
    conn: asyncpg.Connection,
    *,
    limit: int,
) -> list[str]:
    """Return project ids with KB chunks but no insight rows (bounded batch)."""
    rows = await conn.fetch(
        """
        SELECT p.id::text
        FROM projects p
        WHERE p.kb_chunk_count > 0
          AND NOT EXISTS (
              SELECT 1
              FROM project_insights pi
              WHERE pi.project_id = p.id
          )
        ORDER BY p.updated_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [str(row["id"]) for row in rows]


async def list_projects_needing_insight_backfill_for_service(*, limit: int) -> list[str]:
    """Service wrapper for list_projects_needing_insight_backfill."""
    return await _with_service_pool(list_projects_needing_insight_backfill, limit=limit)


async def list_projects_needing_insight_backfill_page(
    conn: asyncpg.Connection,
    *,
    limit: int,
    cursor_updated_at: datetime | None = None,
    cursor_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return one cursor page of projects needing backfill ordered by recency."""
    rows = await conn.fetch(
        """
        SELECT
            p.id::text AS id,
            p.updated_at
        FROM projects p
        WHERE p.kb_chunk_count > 0
          AND NOT EXISTS (
              SELECT 1
              FROM project_insights pi
              WHERE pi.project_id = p.id
          )
          AND (
              $2::timestamptz IS NULL
              OR p.updated_at < $2::timestamptz
              OR (p.updated_at = $2::timestamptz AND p.id < $3::uuid)
          )
        ORDER BY p.updated_at DESC, p.id DESC
        LIMIT $1
        """,
        limit,
        cursor_updated_at,
        cursor_id,
    )
    return [dict(row) for row in rows]


async def list_projects_needing_insight_backfill_page_for_service(
    *,
    limit: int,
    cursor_updated_at: datetime | None = None,
    cursor_id: str | None = None,
) -> list[dict[str, Any]]:
    """Service wrapper for list_projects_needing_insight_backfill_page."""
    return await _with_service_pool(
        list_projects_needing_insight_backfill_page,
        limit=limit,
        cursor_updated_at=cursor_updated_at,
        cursor_id=cursor_id,
    )


async def reset_stuck_generating_reports_for_service() -> int:
    """Reset startup-stale generating rows to pending and return affected count."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            """
            WITH updated AS (
                UPDATE project_insights
                SET full_report_status = 'pending',
                    updated_at = NOW()
                WHERE full_report_status = 'generating'
                RETURNING 1
            )
            SELECT COUNT(*)::int FROM updated
            """
        )
    return int(value or 0)
