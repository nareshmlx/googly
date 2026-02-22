"""Add durable chat_messages table for persistent chat history

Revision ID: 004
Revises: 003
Create Date: 2026-02-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_chat_messages_role",
        ),
    )
    op.create_index(
        "ix_chat_messages_project_session_created",
        "chat_messages",
        ["project_id", "session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_chat_messages_user_project_created",
        "chat_messages",
        ["user_id", "project_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_messages_user_project_created", table_name="chat_messages")
    op.drop_index("ix_chat_messages_project_session_created", table_name="chat_messages")
    op.drop_table("chat_messages")
