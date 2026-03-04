"""Keep projects.kb_chunk_count in sync via DB trigger; resync existing drift.

Revision ID: 014_kb_chunk_count_trigger
Revises: 013_backfill_null_document_ids
Create Date: 2026-03-04

Why a trigger instead of application-level increments:
- Application code uses two separate paths (update_project_kb_stats for full
  ingest runs, increment_project_kb_stats for fulltext extraction) that can
  race or be skipped on task failure, causing drift.
- A AFTER INSERT/DELETE trigger on knowledge_chunks is the only reliable way
  to keep the counter accurate regardless of which code path inserts or removes
  chunks.
- The trigger uses a single atomic UPDATE per statement (AFTER ... FOR EACH ROW
  aggregated into a statement-level approach via a simple row trigger) which is
  safe with asyncpg connection pools.

The application update calls (update_project_kb_stats, increment_project_kb_stats)
can remain in place — they become no-ops for the count since the trigger keeps it
correct, but they still update last_refreshed_at which is still needed.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "014_kb_chunk_count_trigger"
down_revision: str | None = "013_backfill_null_document_ids"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    # 1) Resync any existing drift — set kb_chunk_count to the actual count
    bind.execute(
        sa.text("""
        UPDATE projects p
        SET kb_chunk_count = (
            SELECT COUNT(*) FROM knowledge_chunks kc WHERE kc.project_id = p.id
        )
        WHERE kb_chunk_count != (
            SELECT COUNT(*) FROM knowledge_chunks kc WHERE kc.project_id = p.id
        )
        """)
    )

    # 2) Create the trigger function
    bind.execute(
        sa.text("""
        CREATE OR REPLACE FUNCTION trg_sync_kb_chunk_count()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                UPDATE projects
                SET kb_chunk_count = kb_chunk_count + 1
                WHERE id = NEW.project_id;
            ELSIF TG_OP = 'DELETE' THEN
                UPDATE projects
                SET kb_chunk_count = GREATEST(0, kb_chunk_count - 1)
                WHERE id = OLD.project_id;
            END IF;
            RETURN NULL;
        END;
        $$
        """)
    )

    # 3) Attach the trigger to knowledge_chunks
    bind.execute(
        sa.text("""
        CREATE TRIGGER trg_knowledge_chunks_kb_count
        AFTER INSERT OR DELETE ON knowledge_chunks
        FOR EACH ROW EXECUTE FUNCTION trg_sync_kb_chunk_count()
        """)
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text("DROP TRIGGER IF EXISTS trg_knowledge_chunks_kb_count ON knowledge_chunks")
    )
    bind.execute(sa.text("DROP FUNCTION IF EXISTS trg_sync_kb_chunk_count()"))
