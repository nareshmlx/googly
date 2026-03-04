"""Shared cache + stale-fallback helper for external tool calls."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import structlog

from app.core.cache import TwoTierCache
from app.core.cache_keys import build_stale_cache_key
from app.core.config import settings
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)


async def cached_external_results(
    *,
    cache_key: str,
    fetch_fn: Callable[[], Awaitable[list[dict]]],
    ttl: int,
    stale_event: str,
    stale_context: dict[str, object],
    result_key: str = "results",
) -> list[dict]:
    """Get external results from fresh cache, fetch, or stale fallback."""
    stale_key = build_stale_cache_key(cache_key)
    redis_client = None
    cache = None

    try:
        redis_client = await get_redis()
        cache = TwoTierCache(redis_client)
        cached = await cache.get(cache_key)
        if cached is not None:
            payload = cached.get(result_key)
            return payload if isinstance(payload, list) else []
    except Exception:
        logger.warning("external_cache.read_failed", cache_key=cache_key)

    try:
        fetched = await fetch_fn()
        result = fetched if isinstance(fetched, list) else []
    except Exception:
        logger.exception("external_cache.fetch_failed", cache_key=cache_key)
        result = []

    payload = {result_key: result}
    if result and cache is not None and redis_client is not None:
        try:
            await cache.set(cache_key, payload, ttl=ttl)
            await redis_client.setex(
                stale_key,
                settings.CACHE_TTL_STALE,
                json.dumps(payload),
            )
        except Exception:
            logger.warning("external_cache.write_failed", cache_key=cache_key)

    if result:
        return result

    if redis_client is not None:
        try:
            stale_raw = await redis_client.get(stale_key)
            if stale_raw:
                logger.warning(stale_event, **stale_context)
                parsed = json.loads(stale_raw)
                stale_payload = parsed.get(result_key) if isinstance(parsed, dict) else []
                return stale_payload if isinstance(stale_payload, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
        except Exception:
            logger.warning("external_cache.stale_read_failed", cache_key=cache_key)

    return []
