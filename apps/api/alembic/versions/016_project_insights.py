"""Create project_insights table for AI Discover insight clusters.

Revision ID: 016_project_insights
Revises: 015_kc_doc_id_not_null
Create Date: 2026-03-05
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "016_project_insights"
down_revision: str | None = "015_kc_doc_id_not_null"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "project_insights",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("topic_label", sa.Text(), nullable=False),
        sa.Column("executive_summary", sa.Text(), nullable=False),
        sa.Column(
            "key_findings",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "trend_signal",
            sa.String(length=32),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("contradictions", sa.Text(), nullable=True),
        sa.Column(
            "chunk_ids",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "source_doc_ids",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("cluster_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("full_report", sa.Text(), nullable=True),
        sa.Column(
            "full_report_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "source_type_counts",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "full_report_status IN ('pending', 'generating', 'done', 'failed')",
            name="ck_project_insights_full_report_status",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_project_insights_project_id",
        "project_insights",
        ["project_id"],
        unique=False,
    )

    op.execute("ALTER TABLE project_insights ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY project_isolation ON project_insights
        FOR ALL
        USING (
            project_id = ANY(
                string_to_array(
                    current_setting('app.accessible_projects', true),
                    ','
                )::uuid[]
            )
        )
        WITH CHECK (
            project_id = ANY(
                string_to_array(
                    current_setting('app.accessible_projects', true),
                    ','
                )::uuid[]
            )
        )
        """
    )
    op.execute("ALTER TABLE project_insights FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS project_isolation ON project_insights")
    op.execute("ALTER TABLE project_insights DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_project_insights_project_id", table_name="project_insights")
    op.drop_table("project_insights")
