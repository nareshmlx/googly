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

from dataclasses import dataclass

import structlog

from app.core.utils import build_stable_signature, parse_metadata
from app.kb.chunker import chunk_text
from app.kb.embedder import embed_texts
from app.repositories import knowledge as knowledge_repo

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
    Upsert documents into knowledge_documents, then chunk/embed/link to knowledge_chunks.

    Returns total number of chunks inserted.
    """
    if not documents:
        return 0

    # 1. First, upsert logical documents and get their UUIDs.
    # Ensure source_id is always present to satisfy knowledge_documents schema.
    def _ensure_source_id(doc: RawDocument) -> str:
        explicit = str(doc.source_id or "").strip()
        if explicit:
            return explicit
        signature = build_stable_signature(
            [
                str(doc.project_id),
                str(doc.source),
                str(doc.title or "").strip(),
                doc.content.strip()[:256],
            ]
        )
        return f"generated:{signature[:24]}" if signature else "generated:missing"

    prepared_docs: list[tuple[RawDocument, str, dict]] = []
    for doc in documents:
        normalized_metadata = parse_metadata(doc.metadata)
        prepared_docs.append((doc, _ensure_source_id(doc), normalized_metadata))

    # Group by (project_id, source, source_id) to ensure one document row per source.
    unique_docs: dict[tuple[str, str, str], tuple[RawDocument, str, dict]] = {}
    for doc, source_id, metadata in prepared_docs:
        key = (doc.project_id, doc.source, source_id)
        existing = unique_docs.get(key)
        if existing is None or len(doc.content.strip()) > len(existing[0].content.strip()):
            unique_docs[key] = (doc, source_id, metadata)

    doc_records = list(unique_docs.values())
    doc_map = {}  # (project_id, source, source_id) -> document_id

    upsert_records = [
        {
            "project_id": d.project_id,
            "user_id": d.user_id,
            "source": d.source,
            "source_id": source_id,
            "title": d.title or "",
            "summary": d.content or "",
            "metadata": metadata,
        }
        for d, source_id, metadata in doc_records
    ]
    doc_map = await knowledge_repo.upsert_documents(upsert_records)

    # 2. Extract and embed chunks
    all_chunks = []
    for doc, source_id, metadata in prepared_docs:
        doc_id = doc_map.get((doc.project_id, doc.source, source_id))
        if not doc_id:
            continue

        if not doc.content.strip():
            continue

        chunks = chunk_text(doc.content)
        for i, chunk in enumerate(chunks):
            # For multi-chunk docs, suffix source_id so the unique constraint works
            chunk_source_id = f"{source_id}:{i}"
            chunk_metadata = _normalize_chunk_metadata(metadata, chunk_index=i)
            all_chunks.append(
                {
                    "document_id": doc_id,
                    "project_id": doc.project_id,
                    "user_id": doc.user_id,
                    "source": doc.source,
                    "source_id": chunk_source_id,
                    "title": doc.title,
                    "content": chunk,
                    "metadata": chunk_metadata,
                }
            )

    if not all_chunks:
        return 0

    logger.info("ingester.embedding", chunk_count=len(all_chunks))
    texts: list[str] = [str(c["content"]) for c in all_chunks]
    vectors = await embed_texts(texts)

    # 3. Bulk insert chunks
    inserted = await knowledge_repo.insert_chunks(all_chunks, vectors)

    logger.info("ingester.done", inserted=inserted, total_chunks=len(all_chunks))
    return inserted
