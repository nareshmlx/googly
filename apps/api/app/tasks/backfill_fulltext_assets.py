"""ARQ task for gradually backfilling fulltext assets from existing metadata chunks."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import structlog

from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.db import get_db_pool
from app.kb.ingester import RawDocument
from app.repositories import source_asset as source_asset_repo
from app.services.fulltext_resolver import resolve_fulltext_url

logger = structlog.get_logger(__name__)


def _parse_cursor_ts(value: object) -> datetime:
    """Parse cursor timestamp payload into aware datetime for asyncpg."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    raw = str(value or "").strip()
    if not raw:
        return datetime(1970, 1, 1, tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _coerce_metadata(value: object) -> dict:
    """Normalize metadata payload from asyncpg/jsonb into a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _cursor_scope(project_id: str | None) -> str:
    return project_id or "all"


async def backfill_fulltext_assets(ctx: dict, project_id: str | None = None, limit: int = 100) -> dict:
    """Scan metadata-only chunks and enqueue bounded fulltext enrichment jobs."""
    if not settings.ENABLE_FULLTEXT_BACKFILL:
        return {"scheduled": 0, "reason": "disabled"}

    redis = ctx.get("redis")
    if redis is None:
        return {"scheduled": 0, "reason": "missing_redis"}

    if await redis.get(RedisKeys.FULLTEXT_BACKFILL_PAUSE):
        return {"scheduled": 0, "reason": "paused"}

    scope = _cursor_scope(project_id)
    cursor_key = RedisKeys.FULLTEXT_BACKFILL_CURSOR.format(scope=scope)
    cursor_payload = await redis.get(cursor_key)
    cursor = json.loads(cursor_payload) if isinstance(cursor_payload, str) and cursor_payload else {}

    cursor_ts = _parse_cursor_ts(cursor.get("created_at"))
    cursor_id = str(cursor.get("id") or "00000000-0000-0000-0000-000000000000")

    pool = await get_db_pool()
    where_project = "AND kc.project_id = $3::uuid" if project_id else ""
    limit_placeholder = "$4" if project_id else "$3"
    args: list[object] = [cursor_ts, cursor_id]
    if project_id:
        args.append(project_id)
    args.append(max(1, min(limit, 500)))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT kc.id::text, kc.project_id::text, kc.user_id, kc.source, kc.source_id,
                   kc.title, kc.metadata, kc.created_at
            FROM knowledge_chunks kc
            WHERE kc.source IN ('paper', 'patent')
              AND COALESCE(kc.metadata->>'content_level', 'abstract') != 'fulltext'
              AND (
                    kc.created_at > $1::timestamptz
                    OR (kc.created_at = $1::timestamptz AND kc.id::text > $2)
              )
              {where_project}
            ORDER BY kc.created_at ASC, kc.id::text ASC
            LIMIT {limit_placeholder}
            """,
            *args,
        )

    scheduled = 0
    last_created_at = cursor_ts.isoformat()
    last_id = cursor_id
    for row in rows:
        metadata = _coerce_metadata(row.get("metadata"))
        doc = RawDocument(
            project_id=str(row["project_id"]),
            user_id=str(row["user_id"]),
            source=str(row["source"]),
            source_id=str(row["source_id"] or ""),
            title=str(row["title"] or ""),
            content="",
            metadata=metadata,
        )
        resolved = resolve_fulltext_url(doc)
        if resolved.status != "success" or not resolved.canonical_url or not resolved.resolved_url:
            continue

        asset_id = await source_asset_repo.upsert_source_asset(
            pool,
            project_id=str(row["project_id"]),
            user_id=str(row["user_id"]),
            source=str(row["source"]),
            source_id=str(row["source_id"] or ""),
            title=str(row["title"] or ""),
            source_url=str(metadata.get("url") or resolved.resolved_url),
            resolved_url=resolved.resolved_url,
            canonical_url=resolved.canonical_url,
            source_fetcher=resolved.source_fetcher,
        )
        if asset_id:
            await redis.enqueue_job("ingest_source_asset", asset_id)
            scheduled += 1

        created_at_value = row["created_at"]
        if isinstance(created_at_value, datetime):
            last_created_at = created_at_value.isoformat()
        last_id = str(row["id"])

    await redis.setex(
        cursor_key,
        RedisTTL.FULLTEXT_BACKFILL_CURSOR.value,
        json.dumps({"created_at": last_created_at, "id": last_id}),
    )

    logger.info(
        "fulltext_backfill.done",
        project_id=project_id,
        scanned=len(rows),
        scheduled=scheduled,
    )
    return {"scheduled": scheduled, "scanned": len(rows), "scope": scope}
