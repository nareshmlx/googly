"""Initial schema — production-ready

Revision ID: 001
Revises:
Create Date: 2026-02-18

Schema matches PROJECT_OVERVIEW.md §7 exactly:
- users: clerk_id, tier
- projects: structured_intent JSONB, refresh_strategy with CHECK, last_refreshed_at, kb_chunk_count
- knowledge_chunks: user_id denormalized, source/source_id/title/metadata/expires_at,
  embedding vector(1536), RLS policy, IVFFlat index, UNIQUE(project_id, source, source_id)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # users
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("clerk_id", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("tier", sa.String(50), nullable=False, server_default="free"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_clerk_id", "users", ["clerk_id"], unique=True)

    # projects
    op.create_table(
        "projects",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column(
            "structured_intent",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "refresh_strategy",
            sa.String(50),
            nullable=False,
            server_default="once",
        ),
        sa.Column("last_refreshed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("kb_chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "refresh_strategy IN ('once', 'daily', 'weekly', 'on_demand')",
            name="ck_projects_refresh_strategy",
        ),
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"])
    # Partial index for refresh scheduling — only rows that need periodic refresh
    op.execute(
        """
        CREATE INDEX ix_projects_refresh_due
        ON projects (refresh_strategy, last_refreshed_at)
        WHERE refresh_strategy IN ('daily', 'weekly')
        """
    )

    # knowledge_chunks — vector(1536) requires pgvector
    op.create_table(
        "knowledge_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_id", sa.Text, nullable=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        # vector(1536) — pgvector type; raw DDL because SQLAlchemy doesn't know this type
        sa.Column("embedding", sa.Text, nullable=True),  # placeholder, replaced below
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "source", "source_id", name="uq_kc_project_source"),
    )

    # Drop the placeholder text column and add the real vector column
    op.execute("ALTER TABLE knowledge_chunks DROP COLUMN embedding")
    op.execute("ALTER TABLE knowledge_chunks ADD COLUMN embedding vector(1536)")

    # Indexes on knowledge_chunks
    op.create_index("ix_kc_project_id", "knowledge_chunks", ["project_id"])
    op.create_index("ix_kc_user_id", "knowledge_chunks", ["user_id"])
    op.execute(
        """
        CREATE INDEX ix_kc_expires
        ON knowledge_chunks (expires_at)
        WHERE expires_at IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX ix_kc_metadata
        ON knowledge_chunks USING gin (metadata)
        """
    )
    # IVFFlat vector index — only useful once rows exist; created here for schema completeness
    op.execute(
        """
        CREATE INDEX ix_kc_vector
        ON knowledge_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

    # Row Level Security — multi-tenant isolation
    op.execute("ALTER TABLE knowledge_chunks ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY project_isolation ON knowledge_chunks
        FOR ALL
        USING (
            project_id = ANY(
                string_to_array(
                    current_setting('app.accessible_projects', true),
                    ','
                )::uuid[]
            )
        )
        """
    )
    # Superuser bypass so the ARQ worker can insert without setting RLS
    op.execute("ALTER TABLE knowledge_chunks FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS project_isolation ON knowledge_chunks")
    op.execute("ALTER TABLE knowledge_chunks DISABLE ROW LEVEL SECURITY")
    op.drop_table("knowledge_chunks")
    op.drop_table("projects")
    op.drop_table("users")
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
    op.execute("DROP EXTENSION IF EXISTS vector")
