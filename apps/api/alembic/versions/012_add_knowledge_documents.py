"""add_knowledge_documents

Revision ID: bf6856672f9a
Revises: 011_hnsw_index
Create Date: 2026-03-03 12:29:39.087498

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '012_add_knowledge_documents'
down_revision: str | None = '011_hnsw_index'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create knowledge_documents table
    op.create_table(
        'knowledge_documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('source_id', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), server_default='', nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('metadata', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'source', 'source_id', name='uq_kd_project_source_id')
    )
    op.create_index('ix_kd_project_id', 'knowledge_documents', ['project_id'], unique=False)
    op.create_index('ix_kd_source_id', 'knowledge_documents', ['source_id'], unique=False)

    # 2. Add document_id to knowledge_chunks
    op.add_column('knowledge_chunks', sa.Column('document_id', sa.UUID(), nullable=True))
    op.create_foreign_key('fk_kc_document_id', 'knowledge_chunks', 'knowledge_documents', ['document_id'], ['id'], ondelete='CASCADE')
    op.create_index('ix_kc_document_id', 'knowledge_chunks', ['document_id'], unique=False)

    # 3. Data Migration: Create documents from unique (project_id, source, base_source_id)
    # Preserve source IDs that naturally end with :digits by stripping suffixes only when
    # they are likely ingestion-generated chunk/fulltext suffixes.
    op.execute(sa.text(r"""
        WITH normalized_chunks AS (
            SELECT
                kc.project_id,
                kc.user_id,
                kc.source,
                kc.source_id,
                kc.title,
                kc.created_at,
                REGEXP_REPLACE(
                    REGEXP_REPLACE(kc.source_id, '(\:fulltext)(\:[0-9]+)?$', ''),
                    '(\:[0-9]+)$',
                    ''
                ) AS stripped_id,
                COUNT(*) OVER (
                    PARTITION BY
                        kc.project_id,
                        kc.user_id,
                        kc.source,
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(kc.source_id, '(\:fulltext)(\:[0-9]+)?$', ''),
                            '(\:[0-9]+)$',
                            ''
                        )
                ) AS stripped_group_size
            FROM knowledge_chunks kc
        ),
        resolved_chunks AS (
            SELECT
                project_id,
                user_id,
                source,
                CASE
                    WHEN source_id ~ '(\:fulltext)(\:[0-9]+)?$' OR stripped_group_size > 1
                    THEN stripped_id
                    ELSE source_id
                END AS base_id,
                title,
                created_at
            FROM normalized_chunks
        )
        INSERT INTO knowledge_documents (id, project_id, user_id, source, source_id, title, metadata, created_at, updated_at)
        SELECT
            uuid_generate_v4(),
            project_id,
            user_id,
            source,
            base_id,
            MIN(title),
            '{}'::jsonb,
            MIN(created_at),
            NOW()
        FROM resolved_chunks
        GROUP BY project_id, user_id, source, base_id
        ON CONFLICT (project_id, source, source_id) DO NOTHING
    """).execution_options(no_parameters=True))

    # 4. Link chunks to documents
    op.execute(sa.text(r"""
        WITH normalized_chunks AS (
            SELECT
                kc.id,
                kc.project_id,
                kc.source,
                kc.source_id,
                REGEXP_REPLACE(
                    REGEXP_REPLACE(kc.source_id, '(\:fulltext)(\:[0-9]+)?$', ''),
                    '(\:[0-9]+)$',
                    ''
                ) AS stripped_id,
                COUNT(*) OVER (
                    PARTITION BY
                        kc.project_id,
                        kc.source,
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(kc.source_id, '(\:fulltext)(\:[0-9]+)?$', ''),
                            '(\:[0-9]+)$',
                            ''
                        )
                ) AS stripped_group_size
            FROM knowledge_chunks kc
        ),
        resolved_chunks AS (
            SELECT
                id,
                project_id,
                source,
                CASE
                    WHEN source_id ~ '(\:fulltext)(\:[0-9]+)?$' OR stripped_group_size > 1
                    THEN stripped_id
                    ELSE source_id
                END AS base_id
            FROM normalized_chunks
        )
        UPDATE knowledge_chunks kc
        SET document_id = kd.id
        FROM resolved_chunks rc
        JOIN knowledge_documents kd
          ON kd.project_id = rc.project_id
         AND kd.source = rc.source
         AND kd.source_id = rc.base_id
        WHERE kc.id = rc.id
    """).execution_options(no_parameters=True))


def downgrade() -> None:
    op.drop_index('ix_kc_document_id', table_name='knowledge_chunks')
    op.drop_constraint('fk_kc_document_id', 'knowledge_chunks', type_='foreignkey')
    op.drop_column('knowledge_chunks', 'document_id')
    op.drop_index('ix_kd_source_id', table_name='knowledge_documents')
    op.drop_index('ix_kd_project_id', table_name='knowledge_documents')
    op.drop_table('knowledge_documents')
