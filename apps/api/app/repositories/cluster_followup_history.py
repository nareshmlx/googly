"""Repository functions for per-insight follow-up history persistence."""

from __future__ import annotations

from datetime import datetime
from typing import TypeVar
from uuid import uuid4

import asyncpg

from app.core.db import get_db_pool

T = TypeVar("T")


async def _with_service_pool(fn, *args, **kwargs) -> T:  # type: ignore[no-untyped-def]
    """Run a connection-bound follow-up repository function with managed pool."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await fn(conn, *args, **kwargs)  # type: ignore[no-any-return]


async def append_messages(
    conn: asyncpg.Connection,
    *,
    insight_id: str,
    project_id: str,
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    context_source: str,
) -> None:
    """Insert one user/assistant follow-up turn atomically for an insight thread."""
    async with conn.transaction():
        await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
        user_uuid = str(uuid4())
        assistant_uuid = str(uuid4())
        await conn.execute(
            """
            INSERT INTO cluster_followup_messages
                (id, insight_id, project_id, user_id, role, content, context_source, created_at)
            VALUES
                ($1::uuid, $2::uuid, $3::uuid, $4, 'user', $5, $6, NOW()),
                ($7::uuid, $2::uuid, $3::uuid, $4, 'assistant', $8, $6, NOW())
            """,
            user_uuid,
            insight_id,
            project_id,
            user_id,
            user_msg,
            context_source,
            assistant_uuid,
            assistant_msg,
        )


async def append_messages_for_service(
    *,
    insight_id: str,
    project_id: str,
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    context_source: str,
) -> None:
    """Service wrapper for append_messages."""
    await _with_service_pool(
        append_messages,
        insight_id=insight_id,
        project_id=project_id,
        user_id=user_id,
        user_msg=user_msg,
        assistant_msg=assistant_msg,
        context_source=context_source,
    )


async def get_history(
    conn: asyncpg.Connection,
    *,
    insight_id: str,
    project_id: str,
    user_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Load follow-up history rows ordered newest-first for one insight/user."""
    await conn.execute("SELECT set_config('app.accessible_projects', $1, true)", project_id)
    rows = await conn.fetch(
        """
        SELECT
            id::text,
            role,
            content,
            context_source,
            created_at
        FROM cluster_followup_messages
        WHERE insight_id = $1::uuid
          AND project_id = $2::uuid
          AND user_id = $3
        ORDER BY created_at DESC, id DESC
        LIMIT $4 OFFSET $5
        """,
        insight_id,
        project_id,
        user_id,
        limit,
        offset,
    )
    # Reverse to return chronological order to the caller if needed, 
    # but usually pagination for "history" starts from newest.
    # The requirement (M4) just says "pagination".
    return [dict(row) for row in rows]


async def get_history_for_service(
    *,
    insight_id: str,
    project_id: str,
    user_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Service wrapper for get_history."""
    return await _with_service_pool(
        get_history,
        insight_id=insight_id,
        project_id=project_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )



async def delete_history_older_than(conn: asyncpg.Connection, cutoff: datetime) -> int:
    """Delete old follow-up rows by timestamp and return deleted count."""
    result = await conn.execute(
        """
        DELETE FROM cluster_followup_messages
        WHERE created_at < $1::timestamptz
        """,
        cutoff,
    )
    return int((result or "DELETE 0").split()[-1])


async def delete_history_older_than_for_service(cutoff: datetime) -> int:
    """Delete old follow-up rows with internally managed pool."""
    return await _with_service_pool(delete_history_older_than, cutoff)
