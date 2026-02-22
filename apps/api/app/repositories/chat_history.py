"""ChatHistoryRepository — asyncpg queries for durable chat history."""

from datetime import UTC, datetime, timedelta

import asyncpg


async def verify_session_ownership(
    pool: asyncpg.Pool,
    project_id: str,
    user_id: str,
    session_id: str,
) -> bool:
    """
    Verify that the given session_id is either new or belongs to the specified user.

    Returns False ONLY if the session exists and belongs to a DIFFERENT user.
    Returns True if the session is new (doesn't exist yet) or belongs to user_id.

    This prevents session hijacking while allowing new sessions to be created.

    RLS Note: Queries without RLS scope to detect cross-user session attempts.
    RLS will enforce isolation on actual message reads/writes.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT user_id
            FROM chat_messages
            WHERE project_id = $1::uuid
              AND session_id = $2
            LIMIT 1
            """,
            project_id,
            session_id,
        )

    # Session doesn't exist yet — allow it (will be created)
    if row is None:
        return True

    # Session exists — check if it belongs to current user
    return row["user_id"] == user_id


async def insert_chat_turn(
    pool: asyncpg.Pool,
    project_id: str,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Insert a user+assistant chat turn atomically into Postgres.

    Uses a single INSERT with two VALUES rows so a partial turn is impossible.

    RLS Note: Sets app.accessible_projects before INSERT to enforce RLS WITH CHECK clause.
    Even though worker runs as superuser, this ensures writes respect project boundaries.
    """
    created_at_user = datetime.now(UTC)
    created_at_assistant = created_at_user + timedelta(microseconds=1)

    async with pool.acquire() as conn, conn.transaction():
        # Set RLS scope to enforce WITH CHECK clause on INSERT
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)",
            project_id,
        )
        await conn.execute(
            """
            INSERT INTO chat_messages (project_id, user_id, session_id, role, content, created_at)
            VALUES
                ($1::uuid, $2, $3, 'user', $4, $6),
                ($1::uuid, $2, $3, 'assistant', $5, $7)
            """,
            project_id,
            user_id,
            session_id,
            user_message,
            assistant_message,
            created_at_user,
            created_at_assistant,
        )


async def fetch_chat_history(
    pool: asyncpg.Pool,
    project_id: str,
    user_id: str,
    session_id: str,
) -> list[dict]:
    """
    Fetch ordered chat messages for one user/project/session.

    Ordered by created_at then id to keep stable turn order.

    RLS Note: Sets app.accessible_projects before SELECT to enforce RLS policy.
    """
    async with pool.acquire() as conn, conn.transaction():
        # Set RLS scope before querying
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)",
            project_id,
        )
        rows = await conn.fetch(
            """
            SELECT role, content
            FROM chat_messages
            WHERE project_id = $1::uuid
              AND user_id = $2
              AND session_id = $3
            ORDER BY created_at ASC, id ASC
            """,
            project_id,
            user_id,
            session_id,
        )
    return [dict(r) for r in rows]


async def replace_chat_history(
    pool: asyncpg.Pool,
    project_id: str,
    user_id: str,
    session_id: str,
    messages: list[dict],
) -> None:
    """
    Replace one session history atomically (used for Redis -> DB backfill).

    Message order is preserved by assigning strictly increasing created_at values.
    """
    if not messages:
        return

    base_ts = datetime.now(UTC)
    insert_rows: list[tuple[str, str, str, str, str, datetime]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        insert_rows.append(
            (
                project_id,
                user_id,
                session_id,
                role,
                content,
                base_ts + timedelta(microseconds=idx),
            )
        )

    if not insert_rows:
        return

    async with pool.acquire() as conn, conn.transaction():
        # Set RLS scope before DELETE + INSERT to enforce RLS policy
        await conn.execute(
            "SELECT set_config('app.accessible_projects', $1, true)",
            project_id,
        )
        await conn.execute(
            """
            DELETE FROM chat_messages
            WHERE project_id = $1::uuid
              AND user_id = $2
              AND session_id = $3
            """,
            project_id,
            user_id,
            session_id,
        )
        await conn.executemany(
            """
            INSERT INTO chat_messages
                (project_id, user_id, session_id, role, content, created_at)
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            """,
            insert_rows,
        )
