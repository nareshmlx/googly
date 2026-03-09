"""Create cluster_followup_messages table for per-insight follow-up history.

Revision ID: 017_cluster_followup_messages
Revises: 016_project_insights
Create Date: 2026-03-05
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "017_cluster_followup_messages"
down_revision: str | None = "016_project_insights"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cluster_followup_messages",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("insight_id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context_source", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_cfm_role"),
        sa.CheckConstraint(
            "context_source IN ('cluster', 'cluster_docs_expanded')",
            name="ck_cfm_context_source",
        ),
        sa.ForeignKeyConstraint(["insight_id"], ["project_insights.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cfm_insight_user",
        "cluster_followup_messages",
        ["insight_id", "user_id"],
        unique=False,
    )
    op.create_index(
        "ix_cfm_project_user_created",
        "cluster_followup_messages",
        ["project_id", "user_id", "created_at"],
        unique=False,
    )

    op.execute("ALTER TABLE cluster_followup_messages ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY project_isolation ON cluster_followup_messages
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
    op.execute("ALTER TABLE cluster_followup_messages FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS project_isolation ON cluster_followup_messages")
    op.execute("ALTER TABLE cluster_followup_messages DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_cfm_project_user_created", table_name="cluster_followup_messages")
    op.drop_index("ix_cfm_insight_user", table_name="cluster_followup_messages")
    op.drop_table("cluster_followup_messages")
