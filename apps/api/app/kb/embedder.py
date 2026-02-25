"""KB Embedder — batch OpenAI text-embedding-3-small with Redis caching.

Design decisions:
- One API call for N texts (Rule 7: batch, never one-at-a-time).
- Per-text Redis cache keyed by SHA-256 of the text content. Cache hit means
  zero OpenAI cost for repeated chunks (e.g., refresh runs on unchanged docs).
- Returns vectors in the same order as the input list.
- Empty or whitespace-only strings are rejected early — pgvector cannot store
  a zero-dimension vector and OpenAI will error on blank input.
"""

import hashlib
import json

import structlog
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.constants import EmbeddingBatchSize, RedisKeys, RedisTTL
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Module-level singleton — one connection pool per worker process, reused across
# all embed_texts() calls. Creating a new AsyncOpenAI() per call creates a new
# httpx.AsyncClient on every invocation, which wastes connections under load.
_openai_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return 1536-dim embeddings for each text, using Redis cache to avoid
    re-embedding identical content across requests or refresh runs.

    Raises ValueError if any text is blank.
    Raises on OpenAI API errors — caller must handle (tenacity retry recommended).
    """
    if not texts:
        return []

    for i, t in enumerate(texts):
        if not t.strip():
            raise ValueError(f"Empty text at index {i} — cannot embed blank content")

    redis = await get_redis()
    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    # Include model version in cache key to auto-invalidate when OpenAI updates model
    # Format: embed:{model_version}:{text_hash}
    # When model changes, update EMBEDDING_MODEL_VERSION in config.py
    cache_keys = [
        RedisKeys.EMBED_CACHE.format(
            model_version=settings.EMBEDDING_MODEL_VERSION,
            hash=hashlib.sha256(t.encode()).hexdigest(),
        )
        for t in texts
    ]
    cached_values = await redis.mget(*cache_keys)

    for i, cached in enumerate(cached_values):
        if cached is not None:
            results[i] = json.loads(cached)
        else:
            uncached_indices.append(i)
            uncached_texts.append(texts[i])

    if uncached_indices:
        logger.info(
            "embedder.api_call",
            total=len(texts),
            cached=len(texts) - len(uncached_indices),
            calling_api=len(uncached_indices),
        )
        client = _get_client()

        # Batch in chunks of EmbeddingBatchSize.DEFAULT (100) to stay within API limits
        batch_size = EmbeddingBatchSize.DEFAULT
        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            response = await client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            # OpenAI returns embeddings in the same order as input
            all_vectors.extend([item.embedding for item in response.data])

        # Write to results + cache in one pipeline
        pipe = redis.pipeline()
        for list_pos, orig_idx in enumerate(uncached_indices):
            vector = all_vectors[list_pos]
            results[orig_idx] = vector
            key = cache_keys[orig_idx]
            pipe.setex(key, RedisTTL.EMBED_CACHE.value, json.dumps(vector))
        await pipe.execute()

        logger.info("embedder.done", embedded=len(uncached_indices))

    return results  # type: ignore[return-value]  # all slots filled above
