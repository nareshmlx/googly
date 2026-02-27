"""Create a test project in the database"""

import asyncio
from uuid import UUID

from app.core.db import get_db_pool


async def create_test_project():
    pool = await get_db_pool()

    project_id = UUID("b448e66a-fc8d-405e-bcf9-b9c452fe2a4b")
    user_id = "test-user-123"

    # Check if project exists
    existing = await pool.fetchrow("SELECT id FROM projects WHERE id = $1", project_id)

    if existing:
        print(f"Project {project_id} already exists!")
        return

    # Insert test project
    await pool.execute(
        """
        INSERT INTO projects (
            id, user_id, title, description, created_at, updated_at,
            refresh_strategy, kb_chunk_count, structured_intent,
            instagram_enabled, tiktok_enabled, openalex_enabled
        )
        VALUES ($1, $2, $3, $4, NOW(), NOW(), 'weekly', 0, '{}', true, true, true)
    """,
        project_id,
        user_id,
        "Test Beauty Project",
        "E2E test project for Googly API",
    )

    print(f"Created test project: {project_id} for user: {user_id}")


if __name__ == "__main__":
    asyncio.run(create_test_project())
