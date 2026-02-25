"""Change user_id columns from UUID to TEXT for Clerk compatibility

Revision ID: 002
Revises: 001
Create Date: 2026-02-18

Clerk user IDs are strings like "user_2abc123..." — not UUIDs. The initial
migration defined projects.user_id and knowledge_chunks.user_id as UUID with
a FK to users.id. This migration:

1. Drops the FK constraint on projects.user_id (Clerk IDs are stored
   directly — we do not require a row in the users table for every project).
2. Alters projects.user_id to TEXT.
3. Alters knowledge_chunks.user_id to TEXT (denormalized copy, same issue).

The projects.id and knowledge_chunks.id primary keys remain UUID — only the
owner identifier columns change type.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop the FK constraint that required projects.user_id to reference users.id.
    # We store Clerk user IDs directly — a users table row is not required.
    op.drop_constraint("projects_user_id_fkey", "projects", type_="foreignkey")

    # Alter projects.user_id: UUID → TEXT (with explicit cast via USING clause)
    op.execute("ALTER TABLE projects ALTER COLUMN user_id TYPE TEXT USING user_id::text")

    # Alter knowledge_chunks.user_id: UUID → TEXT (denormalized, same reason)
    op.execute("ALTER TABLE knowledge_chunks ALTER COLUMN user_id TYPE TEXT USING user_id::text")


def downgrade() -> None:
    # Revert knowledge_chunks.user_id to UUID
    op.execute("ALTER TABLE knowledge_chunks ALTER COLUMN user_id TYPE uuid USING user_id::uuid")

    # Revert projects.user_id to UUID and restore the FK
    op.execute("ALTER TABLE projects ALTER COLUMN user_id TYPE uuid USING user_id::uuid")
    op.create_foreign_key(
        "projects_user_id_fkey",
        "projects",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
