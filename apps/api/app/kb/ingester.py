"""KB Ingester — chunk text, embed in batch, bulk upsert to knowledge_chunks.

Design decisions:
- Semantic chunking: fixed max_tokens with overlap to preserve context across chunk boundaries.
  Simple and fast. Recursive semantic splitter adds complexity with marginal gain at this stage.
- One OpenAI embed call per ingest batch (via embedder.embed_texts).
- Bulk asyncpg INSERT ... ON CONFLICT DO NOTHING so repeated ingestion of the same
  source_id is idempotent — safe to re-run on refresh without duplicating chunks.
- RLS note: ingestion runs in the ARQ worker context. The worker connects with the
  superuser (googly) role so FORCE ROW LEVEL SECURITY is bypassed for inserts.
  Reads from the API always set app.accessible_projects before querying.
"""

import json
import uuid
from dataclasses import dataclass

import structlog

from app.core.db import get_db_pool
from app.kb.chunker import chunk_text
from app.kb.embedder import embed_texts

logger = structlog.get_logger(__name__)


@dataclass
class RawDocument:
    """One source document to be chunked and ingested into a project KB."""

    project_id: str
    user_id: str
    source: str  # upload | news | paper | patent | social | search
    source_id: str | None
    title: str | None
    content: str
    metadata: dict  # arbitrary source-specific fields (url, published_at, etc.)


async def ingest_documents(documents: list[RawDocument]) -> int:
    """
    Chunk, embed, and upsert a list of documents into knowledge_chunks.

    Returns total number of chunks inserted (excluding conflicts/duplicates).
    Skips documents with empty content. Skips duplicate (project_id, source, source_id).
    """
    if not documents:
        return 0

    all_chunks: list[tuple[str, str, str, str | None, str | None, str, dict]] = []
    # (project_id, user_id, source, source_id, title, chunk_text, metadata)

    for doc in documents:
        if not doc.content.strip():
            logger.warning("ingester.skip_empty", source=doc.source, source_id=doc.source_id)
            continue

        chunks = chunk_text(doc.content)
        for i, chunk in enumerate(chunks):
            # For multi-chunk docs, suffix source_id so the unique constraint works
            chunk_source_id = f"{doc.source_id}:{i}" if doc.source_id else None
            all_chunks.append(
                (
                    doc.project_id,
                    doc.user_id,
                    doc.source,
                    chunk_source_id,
                    doc.title,
                    chunk,
                    doc.metadata,
                )
            )

    if not all_chunks:
        return 0

    logger.info("ingester.embedding", chunk_count=len(all_chunks))
    texts = [c[5] for c in all_chunks]
    vectors = await embed_texts(texts)

    pool = await get_db_pool()
    inserted = 0

    async with pool.acquire() as conn:
        # Bulk insert — ON CONFLICT DO NOTHING for idempotent re-ingestion
        rows = [
            (
                str(uuid.uuid4()),
                all_chunks[i][0],  # project_id
                all_chunks[i][1],  # user_id
                all_chunks[i][2],  # source
                all_chunks[i][3],  # source_id
                all_chunks[i][4],  # title
                all_chunks[i][5],  # content
                str(vectors[i]),  # embedding as vector literal '[0.1, 0.2, ...]'
                json.dumps(all_chunks[i][6]),  # metadata: dict → JSON string for asyncpg ::jsonb
            )
            for i in range(len(all_chunks))
        ]

        # Insert in batches of 500 to keep individual transactions reasonable.
        batch_size = 500
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            values_sql: list[str] = []
            params: list[object] = []
            for row_index, row in enumerate(batch):
                base = row_index * 9
                values_sql.append(
                    f"(${base + 1}, ${base + 2}, ${base + 3}, ${base + 4}, "
                    f"${base + 5}, ${base + 6}, ${base + 7}, ${base + 8}::vector, ${base + 9}::jsonb)"
                )
                params.extend(row)

            batch_inserted = await conn.fetchval(
                f"""
                WITH inserted AS (
                    INSERT INTO knowledge_chunks
                        (id, project_id, user_id, source, source_id, title, content, embedding, metadata)
                    VALUES {", ".join(values_sql)}
                    ON CONFLICT (project_id, source, source_id) DO NOTHING
                    RETURNING 1
                )
                SELECT COUNT(*)::int FROM inserted
                """,
                *params,
            )
            inserted += int(batch_inserted or 0)

        logger.info("ingester.done", inserted=inserted, total_chunks=len(all_chunks))

    return inserted
