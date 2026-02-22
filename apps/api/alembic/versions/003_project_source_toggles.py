"""Add tiktok_enabled, instagram_enabled, openalex_enabled to projects

Revision ID: 003
Revises: 002
Create Date: 2026-02-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE projects
            ADD COLUMN IF NOT EXISTS tiktok_enabled    BOOLEAN NOT NULL DEFAULT TRUE,
            ADD COLUMN IF NOT EXISTS instagram_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            ADD COLUMN IF NOT EXISTS openalex_enabled  BOOLEAN NOT NULL DEFAULT TRUE
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE projects
            DROP COLUMN IF EXISTS tiktok_enabled,
            DROP COLUMN IF EXISTS instagram_enabled,
            DROP COLUMN IF EXISTS openalex_enabled
    """)
