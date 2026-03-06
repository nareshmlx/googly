"""enriched_description

Revision ID: 96b29dde2bf7
Revises: 015_kc_doc_id_not_null
Create Date: 2026-03-06 13:00:00.820363

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '016_enriched_description'
down_revision: Union[str, None] = '015_kc_doc_id_not_null'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('enriched_description', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'enriched_description')
