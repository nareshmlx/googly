"""Add project social source toggles for YouTube, Reddit, and X.

Revision ID: 008
Revises: 007
Create Date: 2026-02-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add per-project social source toggles for new platform ingesters."""
    op.add_column(
        "projects",
        sa.Column("youtube_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "projects",
        sa.Column("reddit_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "projects",
        sa.Column("x_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )


def downgrade() -> None:
    """Drop social source toggles."""
    op.drop_column("projects", "x_enabled")
    op.drop_column("projects", "reddit_enabled")
    op.drop_column("projects", "youtube_enabled")
