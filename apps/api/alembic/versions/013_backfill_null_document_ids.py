"""Backfill missing knowledge_chunks.document_id links.

Revision ID: 013_backfill_null_document_ids
Revises: 012_add_knowledge_documents
Create Date: 2026-03-04

Both steps use the same CTE + raw SQL pattern as migration 012:
- source_id normalization is computed once per chunk in a CTE and referenced by
  alias — never duplicated in GROUP BY or WHERE clauses.
- The UPDATE matches on kc.id (primary key) so each chunk is linked to exactly
  one document with no cross-join ambiguity.

Normalization rule (identical to 012):
  Strip trailing :fulltext[:N] or :[N] suffixes only when the source_id actually
  matches that pattern — preserving all other IDs unchanged.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013_backfill_null_document_ids"
down_revision: str | None = "012_add_knowledge_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Regex used in both steps — defined once here so they stay in sync.
_SUFFIX_PATTERN = r"(:fulltext)(:[0-9]+)?$|:[0-9]+$"
_STRIP_FULLTEXT = r"(:fulltext)(:[0-9]+)?$"
_STRIP_CHUNK_IDX = r"(:[0-9]+)$"


def upgrade() -> None:
    bind = op.get_bind()

    # 1) Insert missing knowledge_documents for chunks where document_id is null.
    #    CTE computes base_id once per chunk; GROUP BY references the alias so the
    #    REGEXP_REPLACE expression is never duplicated.
    bind.exec_driver_sql(
        r"""
        WITH resolved AS (
            SELECT
                kc.project_id,
                kc.user_id,
                kc.source,
                kc.title,
                kc.created_at,
                CASE
                    WHEN kc.source_id ~ '(:fulltext)(:[0-9]+)?$|:[0-9]+$'
                    THEN REGEXP_REPLACE(
                             REGEXP_REPLACE(kc.source_id, '(:fulltext)(:[0-9]+)?$', ''),
                             '(:[0-9]+)$', ''
                         )
                    ELSE kc.source_id
                END AS base_id
            FROM knowledge_chunks kc
            WHERE kc.document_id IS NULL
        )
        INSERT INTO knowledge_documents
            (id, project_id, user_id, source, source_id, title, summary, metadata, created_at, updated_at)
        SELECT
            uuid_generate_v4(),
            project_id,
            user_id,
            source,
            base_id,
            MIN(title),
            NULL,
            '{}'::jsonb,
            MIN(created_at),
            NOW()
        FROM resolved
        GROUP BY project_id, user_id, source, base_id
        ON CONFLICT (project_id, source, source_id) DO NOTHING
    """
    )

    # 2) Link all null-document chunks to their document rows.
    #    CTE computes base_id once per chunk; outer UPDATE matches on kc.id
    #    (primary key) — exactly one document per chunk, no ambiguity.
    bind.exec_driver_sql(
        r"""
        WITH resolved AS (
            SELECT
                kc.id AS chunk_id,
                kc.project_id,
                kc.source,
                CASE
                    WHEN kc.source_id ~ '(:fulltext)(:[0-9]+)?$|:[0-9]+$'
                    THEN REGEXP_REPLACE(
                             REGEXP_REPLACE(kc.source_id, '(:fulltext)(:[0-9]+)?$', ''),
                             '(:[0-9]+)$', ''
                         )
                    ELSE kc.source_id
                END AS base_id
            FROM knowledge_chunks kc
            WHERE kc.document_id IS NULL
        )
        UPDATE knowledge_chunks kc
        SET document_id = kd.id
        FROM resolved r
        JOIN knowledge_documents kd
          ON kd.project_id = r.project_id
         AND kd.source     = r.source
         AND kd.source_id  = r.base_id
        WHERE kc.id = r.chunk_id
    """
    )


def downgrade() -> None:
    # Data backfill migration: no safe automatic rollback.
    pass
