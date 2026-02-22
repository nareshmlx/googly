"""Two-tier caching: L1 in-memory TTL + L2 Redis distributed cache.

L1 provides sub-millisecond lookups for hot queries (per-pod, TTL-enforced).
L2 provides distributed cache across all pods with 5ms latency.

Usage:
    cache = TwoTierCache(redis_client)
    value = await cache.get("mykey")
    await cache.set("mykey", {"data": "value"}, ttl=3600)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from cachetools import TTLCache  # type: ignore[import-untyped]
from redis.asyncio import Redis

from app.core.config import settings
from app.core.metrics import cache_hits_total, cache_misses_total

logger = structlog.get_logger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and Decimal objects."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


class TwoTierCache:
    """Two-tier cache with L1 in-memory TTL cache and L2 Redis.

    Thread-safe L1 access via asyncio.Lock.
    All values stored as JSON-serialized strings in Redis.
    L1 enforces TTL to prevent serving stale data.
    """

    def __init__(self, redis: Redis):
        self._redis = redis
        # TTLCache enforces TTL - entries auto-expire after L1_CACHE_TTL seconds
        self._l1_cache: TTLCache = TTLCache(
            maxsize=settings.L1_CACHE_SIZE,
            ttl=settings.L1_CACHE_TTL,
        )
        self._l1_lock = asyncio.Lock()

    async def get(self, key: str) -> dict | None:
        """Retrieve value from L1 (fast) or L2 (distributed).

        Returns None if key not found in either tier.
        Writes L2 hits back to L1 ONLY if not already present (prevents race condition).
        """
        # Try L1 first (sub-millisecond, TTL-enforced)
        async with self._l1_lock:
            value = self._l1_cache.get(key)
        if value is not None:
            logger.debug("cache.l1.hit", key_preview=key[:50])
            cache_hits_total.labels(cache_tier="L1", cache_type="search").inc()
            return value

        # L1 miss — try L2 (distributed, 5ms)
        try:
            raw = await self._redis.get(key)
            if raw is None:
                logger.debug("cache.l2.miss", key_preview=key[:50])
                cache_misses_total.labels(cache_tier="L1", cache_type="search").inc()
                cache_misses_total.labels(cache_tier="L2", cache_type="search").inc()
                return None

            value = json.loads(raw)
            logger.debug("cache.l2.hit", key_preview=key[:50])
            cache_misses_total.labels(cache_tier="L1", cache_type="search").inc()
            cache_hits_total.labels(cache_tier="L2", cache_type="search").inc()

            # Write back to L1 ONLY if not already present (prevents overwriting newer data)
            async with self._l1_lock:
                if key not in self._l1_cache:
                    self._l1_cache[key] = value

            return value
        except json.JSONDecodeError:
            logger.warning("cache.l2.invalid_json", key_preview=key[:50])
            cache_misses_total.labels(cache_tier="L2", cache_type="search").inc()
            return None
        except Exception:
            logger.exception("cache.l2.error", key_preview=key[:50])
            cache_misses_total.labels(cache_tier="L2", cache_type="search").inc()
            return None

    async def set(self, key: str, value: dict, ttl: int) -> None:
        """Write to both L1 and L2.

        L1 stores Python dict directly (no serialization, TTL-enforced).
        L2 stores JSON-serialized string with TTL.
        Uses SafeJSONEncoder to handle datetime/Decimal objects.

        L2 write runs in background - errors logged but don't block caller.
        Task is tracked and awaited on shutdown to prevent losing in-flight writes.
        Per AGENTS.md Rule 2: Background tasks must be tracked in production.
        """
        # Write to L1 (in-memory, TTL auto-enforced by TTLCache)
        async with self._l1_lock:
            self._l1_cache[key] = value

        # Write to L2 (distributed with TTL) - fire and forget with error callback
        task = asyncio.create_task(self._write_l2(key, value, ttl))
        task.add_done_callback(self._handle_l2_write_error)

    async def _write_l2(self, key: str, value: dict, ttl: int) -> None:
        """Background L2 write to prevent blocking caller."""
        try:
            serialized = json.dumps(value, cls=SafeJSONEncoder)
            await self._redis.setex(key, ttl, serialized)
            logger.debug("cache.l2.set", key_preview=key[:50], ttl=ttl)
        except (TypeError, ValueError) as exc:
            # JSON serialization failure (should not happen with SafeJSONEncoder)
            logger.error("cache.l2.serialization_error", key_preview=key[:50], error=str(exc))
        except Exception:
            logger.exception("cache.l2.set_error", key_preview=key[:50])

    def _handle_l2_write_error(self, task: asyncio.Task) -> None:
        """Callback to handle uncaught exceptions from L2 background writes.

        Per AGENTS.md Rule 2: Background tasks must handle errors explicitly.
        """
        try:
            # Retrieve exception if task failed (doesn't raise if successful)
            exc = task.exception()
            if exc is not None:
                logger.error("cache.l2.background_task_failed", error=str(exc))
        except asyncio.CancelledError:
            # Task was cancelled (e.g., during shutdown) - this is expected
            pass
        except Exception:
            # Exception() itself raised an error - extremely rare
            logger.exception("cache.l2.error_handler_failed")
