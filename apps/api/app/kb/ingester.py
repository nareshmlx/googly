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


def _normalize_chunk_metadata(metadata: dict, *, chunk_index: int) -> dict:
    """Normalize chunk metadata contract for abstract/fulltext retrieval/ranking."""
    normalized = dict(metadata or {})
    content_level = str(normalized.get("content_level") or "abstract").strip().lower()
    if content_level not in {"abstract", "fulltext"}:
        content_level = "abstract"
    normalized["content_level"] = content_level
    normalized.setdefault("asset_id", None)
    normalized.setdefault("section", "document")
    normalized.setdefault("source_fetcher", str(normalized.get("tool") or "metadata_ingest"))

    page_start = normalized.get("page_start")
    page_end = normalized.get("page_end")
    if isinstance(page_start, int) and isinstance(page_end, int):
        normalized["page_start"] = page_start
        normalized["page_end"] = page_end
    else:
        normalized["page_start"] = chunk_index + 1 if content_level == "fulltext" else None
        normalized["page_end"] = chunk_index + 1 if content_level == "fulltext" else None
    return normalized


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
            chunk_metadata = _normalize_chunk_metadata(doc.metadata, chunk_index=i)
            all_chunks.append(
                (
                    doc.project_id,
                    doc.user_id,
                    doc.source,
                    chunk_source_id,
                    doc.title,
                    chunk,
                    chunk_metadata,
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
        # Build parallel column arrays for unnest bulk INSERT.
        # Embeddings are passed as text (json-formatted list strings) and cast
        # to vector inside SQL — asyncpg's binary COPY protocol has no encoder
        # for pgvector's vector type (OID varies per install), so copy_records_to_table
        # raises InternalClientError: no binary format encoder for type vector.
        ids = [str(uuid.uuid4()) for _ in all_chunks]
        project_ids = [c[0] for c in all_chunks]
        user_ids = [c[1] for c in all_chunks]
        sources = [c[2] for c in all_chunks]
        source_ids = [c[3] for c in all_chunks]
        titles = [c[4] for c in all_chunks]
        contents = [c[5] for c in all_chunks]
        embeddings = [str(vectors[i]) for i in range(len(all_chunks))]
        metadatas = [json.dumps(c[6]) for c in all_chunks]

        batch_inserted = await conn.fetchval(
            """
            WITH inserted AS (
                INSERT INTO knowledge_chunks
                    (id, project_id, user_id, source, source_id, title,
                     content, embedding, metadata)
                SELECT
                    unnest($1::uuid[]),
                    unnest($2::uuid[]),
                    unnest($3::uuid[]),
                    unnest($4::text[]),
                    unnest($5::text[]),
                    unnest($6::text[]),
                    unnest($7::text[]),
                    unnest($8::text[])::vector,
                    unnest($9::text[])::jsonb
                ON CONFLICT (project_id, source, source_id) DO NOTHING
                RETURNING 1
            )
            SELECT COUNT(*)::int FROM inserted
            """,
            ids,
            project_ids,
            user_ids,
            sources,
            source_ids,
            titles,
            contents,
            embeddings,
            metadatas,
        )
        inserted = int(batch_inserted or 0)

    logger.info("ingester.done", inserted=inserted, total_chunks=len(all_chunks))

    return inserted
