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

import asyncio
from dataclasses import dataclass

import httpx
import structlog

from app.core.config import settings
from app.core.openai_client import get_openai_client, openai_completions_with_circuit_breaker
from app.core.retry import retry_with_backoff
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


def _build_chunk_source_id(source_id: str, metadata: dict, *, chunk_index: int) -> str:
    """Return a chunk source ID that keeps fulltext and abstract variants distinct."""
    if str(metadata.get("content_level") or "abstract").strip().lower() == "fulltext":
        return f"{source_id}:fulltext:{chunk_index}"
    return f"{source_id}:{chunk_index}"


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


def _select_summary_excerpts(chunks: list[str], *, max_excerpts: int = 3) -> list[str]:
    """Pick representative excerpts (head/middle/tail) for higher-quality short summaries."""
    cleaned = [str(chunk or "").strip() for chunk in chunks if str(chunk or "").strip()]
    if not cleaned:
        return []
    if len(cleaned) <= max_excerpts:
        return cleaned[:max_excerpts]

    candidate_indices = [0, len(cleaned) // 2, len(cleaned) - 1]
    selected: list[str] = []
    seen_indices: set[int] = set()
    for idx in candidate_indices:
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        selected.append(cleaned[idx])
        if len(selected) >= max_excerpts:
            break
    return selected[:max_excerpts]


async def _summarize_documents(chunks_by_doc: dict[str, list[str]]) -> dict[str, str | None]:
    """Generate concise source-level summaries from top document chunks."""
    if not chunks_by_doc:
        return {}

    client = get_openai_client()
    semaphore = asyncio.Semaphore(settings.DOC_SUMMARIZE_BATCH_SIZE)

    async def _summarize_one(doc_id: str, chunks: list[str]) -> tuple[str, str | None]:
        if not chunks:
            return doc_id, None

        last_failure: Exception | None = None
        last_status_code: int | None = None

        prompt_chunks = "\n\n".join(
            f"Excerpt {idx + 1}:\n{chunk.strip()[:1800]}" for idx, chunk in enumerate(chunks)
        )
        prompt = (
            "Write a high-signal source summary grounded only in the excerpts.\n"
            "Requirements:\n"
            "- 3 to 5 short sentences.\n"
            f"- {settings.DOC_SUMMARIZE_TARGET_MIN_WORDS} to "
            f"{settings.DOC_SUMMARIZE_TARGET_MAX_WORDS} words total.\n"
            "- Plain text only (no bullets, no markdown).\n"
            "- Include concrete actors, numbers, and timing when present.\n"
            "- Start directly with the core topic or claim.\n"
            "- Do not use opener phrases like 'This article', 'The paper', 'This source', or author bylines.\n"
            "- Do not add information not present in excerpts.\n\n"
            f"{prompt_chunks}"
        )

        async def _call_llm() -> str:
            nonlocal last_failure, last_status_code
            async with semaphore:
                try:
                    create_comp = openai_completions_with_circuit_breaker(client)
                    response = await create_comp(
                        model=settings.SUMMARY_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You summarize source documents for executive research cards. "
                                    "Be precise, compact, evidence-grounded, and direct. "
                                    "Return plain text only."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=settings.DOC_SUMMARIZE_MAX_TOKENS,
                    )
                except Exception as exc:
                    last_failure = exc
                    last_status_code = (
                        exc.response.status_code
                        if isinstance(exc, httpx.HTTPStatusError)
                        else None
                    )
                    raise
            return (response.choices[0].message.content or "").strip()

        text = await retry_with_backoff(
            _call_llm,
            max_attempts=3,
            base_delay=1.0,
            max_delay=4.0,
        )
        if text is None:
            logger.warning(
                "ingester.summary.failed",
                doc_id=doc_id,
                model=settings.SUMMARY_MODEL,
                excerpt_count=len(chunks),
                content_chars=sum(len(str(chunk or "")) for chunk in chunks),
                error_type=type(last_failure).__name__ if last_failure else None,
                status_code=last_status_code,
            )
        return doc_id, (text or None)

    pairs = await asyncio.gather(
        *[_summarize_one(doc_id, chunks) for doc_id, chunks in chunks_by_doc.items()]
    )
    return dict(pairs)


async def ingest_documents(documents: list[RawDocument], *, overwrite: bool = False) -> int:
    """
    Upsert documents into knowledge_documents, then chunk/embed/link to knowledge_chunks.

    When overwrite=False (default) chunk inserts use DO NOTHING — safe for repeated
    ingest of the same source without duplicating or altering existing chunks.

    When overwrite=True chunk inserts use DO UPDATE — replaces content and embedding
    on conflict so enriched content (e.g. Gemini transcript) overwrites the original
    short description written during the first ingest pass.

    Returns total number of chunks inserted (or updated when overwrite=True).
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
            "summary": None,
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
        for i, chunk_text_value in enumerate(chunks):
            chunk_metadata = _normalize_chunk_metadata(metadata, chunk_index=i)
            chunk_source_id = _build_chunk_source_id(source_id, chunk_metadata, chunk_index=i)
            all_chunks.append(
                {
                    "document_id": doc_id,
                    "project_id": doc.project_id,
                    "user_id": doc.user_id,
                    "source": doc.source,
                    "source_id": chunk_source_id,
                    "title": doc.title,
                    "content": chunk_text_value,
                    "metadata": chunk_metadata,
                }
            )

    if not all_chunks:
        return 0

    logger.info("ingester.embedding", chunk_count=len(all_chunks))
    texts: list[str] = [str(c["content"]) for c in all_chunks]
    vectors = await embed_texts(texts)

    # 3. Bulk insert chunks
    inserted = await knowledge_repo.insert_chunks(all_chunks, vectors, overwrite=overwrite)

    # 4. Optional one-time source-level summarization (owned by LLM path only).
    if settings.DOC_SUMMARIZE_ENABLED:
        doc_ids = sorted(set(doc_map.values()))
        if doc_ids:
            pending_doc_ids = await knowledge_repo.list_documents_without_summary_for_service(
                doc_ids=doc_ids
            )
            if pending_doc_ids:
                chunks_by_doc: dict[str, list[str]] = {}
                for chunk in all_chunks:
                    doc_id = str(chunk.get("document_id") or "").strip()
                    if not doc_id or doc_id not in pending_doc_ids:
                        continue
                    bucket = chunks_by_doc.setdefault(doc_id, [])
                    bucket.append(str(chunk.get("content") or ""))

                selected_chunks_by_doc: dict[str, list[str]] = {}
                for doc_id, chunk_values in chunks_by_doc.items():
                    excerpts = _select_summary_excerpts(chunk_values, max_excerpts=3)
                    if excerpts:
                        selected_chunks_by_doc[doc_id] = excerpts

                summaries = await _summarize_documents(selected_chunks_by_doc)
                filtered_summaries = {
                    doc_id: summary
                    for doc_id, summary in summaries.items()
                    if summary and doc_id in pending_doc_ids
                }
                updated = 0
                if filtered_summaries:
                    updated = await knowledge_repo.update_document_summaries_for_service(
                        summaries=filtered_summaries
                    )
                logger.info(
                    "ingester.summary.done",
                    requested=len(selected_chunks_by_doc),
                    updated=updated,
                )

    # Log source_ids when batch is small (single-doc enrichment re-ingest) so
    # operators can tie "N chunks inserted" to a specific video/document source_id.
    batch_source_ids = [sid for _, sid, _ in doc_records]
    logger.info(
        "ingester.done",
        inserted=inserted,
        total_chunks=len(all_chunks),
        source_ids=batch_source_ids if len(batch_source_ids) <= 5 else None,
        doc_count=len(batch_source_ids),
    )
    return inserted
