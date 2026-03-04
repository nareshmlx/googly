"""Enforce NOT NULL on knowledge_chunks.document_id.

Revision ID: 015_knowledge_chunks_document_id_not_null
Revises: 014_kb_chunk_count_trigger
Create Date: 2026-03-04

Why this change:
- Migration 013 already back-filled all NULL document_id values by linking
  chunks to their parent knowledge_documents rows (or creating synthetic ones).
- Enforcing NOT NULL at the DB level prevents future bugs where a code path
  inserts a chunk without linking it to a document, which would break the
  document-level deduplication and retrieval logic.
- Safe to run: the pre-check confirms zero NULL rows before altering the
  column. If nulls are somehow present, the migration aborts with a clear error
  rather than silently succeeding with an invalid constraint.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "015_kc_doc_id_not_null"
down_revision: str | None = "014_kb_chunk_count_trigger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    # Safety pre-check: abort if any NULLs remain so the constraint is safe.
    result = bind.execute(
        sa.text("SELECT COUNT(*) FROM knowledge_chunks WHERE document_id IS NULL")
    )
    null_count = result.scalar()
    if null_count:
        raise RuntimeError(
            f"Cannot enforce NOT NULL on knowledge_chunks.document_id: "
            f"{null_count} rows still have NULL document_id. "
            "Run migration 013 again or fix the orphaned chunks first."
        )

    op.alter_column(
        "knowledge_chunks",
        "document_id",
        existing_type=sa.UUID(),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "knowledge_chunks",
        "document_id",
        existing_type=sa.UUID(),
        nullable=True,
    )
