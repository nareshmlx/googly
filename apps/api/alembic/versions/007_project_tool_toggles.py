"""Add project tool toggles and intent embedding.

Revision ID: 007
Revises: 006
Create Date: 2026-02-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rename tool toggle and add new source toggles + project intent embedding."""
    op.alter_column("projects", "openalex_enabled", new_column_name="papers_enabled")
    op.add_column(
        "projects",
        sa.Column("perigon_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "projects",
        sa.Column("tavily_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "projects",
        sa.Column("exa_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "projects",
        sa.Column("patents_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column("projects", sa.Column("intent_embedding", Vector(dim=1536), nullable=True))


def downgrade() -> None:
    """Revert project tool toggles and intent embedding additions."""
    op.drop_column("projects", "intent_embedding")
    op.drop_column("projects", "patents_enabled")
    op.drop_column("projects", "exa_enabled")
    op.drop_column("projects", "tavily_enabled")
    op.drop_column("projects", "perigon_enabled")
    op.alter_column("projects", "papers_enabled", new_column_name="openalex_enabled")
