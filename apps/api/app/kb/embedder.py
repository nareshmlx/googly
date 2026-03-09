"""KB Embedder — batch OpenAI text-embedding-3-small with Redis caching.

Design decisions:
- One API call for N texts (Rule 7: batch, never one-at-a-time).
- Per-text Redis cache keyed by SHA-256 of the text content. Cache hit means
  zero OpenAI cost for repeated chunks (e.g., refresh runs on unchanged docs).
- Returns vectors in the same order as the input list.
- Empty or whitespace-only strings are rejected early — pgvector cannot store
  a zero-dimension vector and OpenAI will error on blank input.
"""

import asyncio
import hashlib

import orjson
import structlog

from app.core.config import settings
from app.core.constants import EMBEDDING_DIM, EmbeddingBatchSize, RedisKeys, RedisTTL
from app.core.openai_client import get_openai_client
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"



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

    redis = None
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
    cached_values: list[str | None] = [None] * len(cache_keys)
    try:
        redis = await get_redis()
        cached_values = await redis.mget(*cache_keys)
    except Exception as exc:
        logger.warning("embedder.cache_read_failed", error=str(exc), text_count=len(texts))

    for i, cached in enumerate(cached_values):
        if cached is not None:
            try:
                results[i] = orjson.loads(cached)
            except Exception:
                # Corrupted / old binary-format entry — treat as cache miss
                uncached_indices.append(i)
                uncached_texts.append(texts[i])
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
        client = get_openai_client()

        # Batch in chunks of EmbeddingBatchSize.DEFAULT (100) to stay within API limits
        batch_size = EmbeddingBatchSize.DEFAULT

        async def _embed_via_openai(batch_texts: list[str]) -> list[list[float]]:
            """Call OpenAI directly when cache/rate-limit state is unavailable."""
            response = await client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
            vectors = [item.embedding for item in response.data]
            if vectors and len(vectors[0]) != EMBEDDING_DIM:
                raise ValueError(
                    f"unexpected_embedding_dim:{len(vectors[0])} expected:{EMBEDDING_DIM}"
                )
            return vectors

        # Win #4: Parallelize OpenAI batches with Global Redis Fixed Window Limit
        async def _embed_batch(batch_texts: list[str]) -> list[list[float]]:
            import time

            if redis is None:
                return await _embed_via_openai(batch_texts)

            while True:
                try:
                    minute = int(time.time() / 60)
                    key = f"sys:ratelimit:embed:{minute}"
                    current_count = int(await redis.get(key) or 0)
                    if current_count >= 2500:
                        await asyncio.sleep(1.0)
                        continue
                    count = await redis.incr(key)
                    if count == 1:
                        await redis.expire(key, 120)
                except Exception as exc:
                    logger.warning(
                        "embedder.rate_limit_cache_unavailable",
                        error=str(exc),
                        batch_size=len(batch_texts),
                    )
                    return await _embed_via_openai(batch_texts)

                # OpenAI Tier 1 limit is roughly 3000 requests per minute
                if count <= 2500:
                    return await _embed_via_openai(batch_texts)

                await asyncio.sleep(1.0)

        batches = [
            uncached_texts[i : i + batch_size] for i in range(0, len(uncached_texts), batch_size)
        ]

        batch_results = await asyncio.gather(*[_embed_batch(b) for b in batches])
        all_vectors = [vec for sublist in batch_results for vec in sublist]

        # Win #3: Binary storage in Redis
        for list_pos, orig_idx in enumerate(uncached_indices):
            results[orig_idx] = all_vectors[list_pos]

        if redis is not None:
            try:
                pipe = redis.pipeline()
                for list_pos, orig_idx in enumerate(uncached_indices):
                    vector = all_vectors[list_pos]
                    key = cache_keys[orig_idx]
                    # Store as JSON string — compatible with decode_responses=True Redis client.
                    # Binary struct.pack is NOT compatible because Redis client decodes all
                    # responses as UTF-8, which crashes on float bytes ≥ 0x80.
                    pipe.setex(key, RedisTTL.EMBED_CACHE, orjson.dumps(vector).decode())
                await pipe.execute()
            except Exception as exc:
                logger.warning("embedder.cache_write_failed", error=str(exc), text_count=len(texts))

        logger.info("embedder.done", embedded=len(uncached_indices))

    return results  # type: ignore[return-value]
