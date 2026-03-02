"""Switch from IVFFlat to HNSW index for knowledge_chunks.

Revision ID: 010
Revises: 009
Create Date: 2026-02-27

HNSW is significantly faster and more accurate than IVFFlat as results grow,
and it does not need to be refitted.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "011_hnsw_index"
down_revision: str | None = "010_fulltext_source_assets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Drop existing IVFFlat index
    op.execute("DROP INDEX IF EXISTS ix_kc_vector")

    # 2. Create HNSW index
    # m=16, ef_construction=64 are good defaults for 1536-dim vectors
    op.execute(
        """
        CREATE INDEX ix_kc_vector_hnsw
        ON knowledge_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    # 1. Drop HNSW
    op.execute("DROP INDEX IF EXISTS ix_kc_vector_hnsw")

    # 2. Recreate IVFFlat
    op.execute(
        """
        CREATE INDEX ix_kc_vector
        ON knowledge_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )
