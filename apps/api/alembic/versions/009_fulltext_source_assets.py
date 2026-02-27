"""Add knowledge_source_assets table for fulltext enrichment tracking.

Revision ID: 010_fulltext_source_assets
Revises: 009_projects_metadata_jsonb
Create Date: 2026-02-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "010_fulltext_source_assets"
down_revision: str | None = "009_projects_metadata_jsonb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create fulltext source asset tracking table and indexes."""
    op.create_table(
        "knowledge_source_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("resolved_url", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("asset_type", sa.String(length=32), nullable=False, server_default="pdf"),
        sa.Column("mime_type", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("blob_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("byte_size", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("fetch_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("extract_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_attempt_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("extracted_chars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("extracted_pages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "source",
            "source_id",
            "canonical_url",
            name="uq_ksa_project_source_sourceid_canonical",
        ),
    )
    op.create_index("ix_ksa_fetch_status", "knowledge_source_assets", ["fetch_status"], unique=False)
    op.create_index("ix_ksa_next_attempt_at", "knowledge_source_assets", ["next_attempt_at"], unique=False)
    op.create_index("ix_ksa_project_source", "knowledge_source_assets", ["project_id", "source"], unique=False)


def downgrade() -> None:
    """Drop fulltext source asset tracking table."""
    op.drop_index("ix_ksa_project_source", table_name="knowledge_source_assets")
    op.drop_index("ix_ksa_next_attempt_at", table_name="knowledge_source_assets")
    op.drop_index("ix_ksa_fetch_status", table_name="knowledge_source_assets")
    op.drop_table("knowledge_source_assets")
