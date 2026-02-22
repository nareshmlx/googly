"""Enable RLS on chat_messages table for multi-tenant isolation

Revision ID: 005
Revises: 004
Create Date: 2026-02-21
"""

from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Enable Row Level Security on chat_messages table.

    This ensures multi-tenant isolation: users can only access messages from
    projects they have access to. The policy uses the app.accessible_projects
    session variable (set before each query) to filter rows.

    Pattern matches knowledge_chunks RLS policy for consistency.
    """
    # Enable RLS on the table
    op.execute("ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY")

    # Create the isolation policy
    # USING clause: filters SELECT queries
    # WITH CHECK clause: validates INSERT/UPDATE operations
    op.execute(
        """
        CREATE POLICY project_isolation ON chat_messages
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

    # Force RLS even for superuser (belt-and-suspenders security)
    # This ensures even the ARQ worker respects RLS for reads
    op.execute("ALTER TABLE chat_messages FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    """Remove RLS from chat_messages table."""
    op.execute("DROP POLICY IF EXISTS project_isolation ON chat_messages")
    op.execute("ALTER TABLE chat_messages DISABLE ROW LEVEL SECURITY")
