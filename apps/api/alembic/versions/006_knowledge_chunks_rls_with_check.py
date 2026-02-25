"""Add WITH CHECK clause to knowledge_chunks RLS policy

Revision ID: 006
Revises: 005
Create Date: 2026-02-21
"""

from collections.abc import Sequence

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add WITH CHECK clause to knowledge_chunks RLS policy.

    The original policy only had USING clause (for SELECT), but was missing
    WITH CHECK clause for INSERT/UPDATE/DELETE validation. This ensures writes
    cannot bypass project isolation even if app.accessible_projects is not set.
    """
    # Drop the incomplete policy
    op.execute("DROP POLICY IF EXISTS project_isolation ON knowledge_chunks")

    # Recreate with both USING and WITH CHECK clauses
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


def downgrade() -> None:
    """Revert to original policy (USING only, no WITH CHECK)."""
    op.execute("DROP POLICY IF EXISTS project_isolation ON knowledge_chunks")

    # Recreate original incomplete policy
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
