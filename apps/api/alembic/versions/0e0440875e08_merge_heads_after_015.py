"""merge parallel heads after 015

Revision ID: 0e0440875e08
Revises: 016_enriched_description, 017_cluster_followup_messages
Create Date: 2026-03-09 11:49:29.776657

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0e0440875e08"
down_revision: tuple[str, str] = ("016_enriched_description", "017_cluster_followup_messages")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Merge the parallel migration branches after revision 015."""


def downgrade() -> None:
    """Unmerge the branches without changing schema state."""
