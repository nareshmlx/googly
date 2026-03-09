"""Circuit breakers for external API calls using async distributed Redis state.

Prevents cascading failures by failing fast when APIs are unhealthy.
State machine: CLOSED (healthy) → OPEN (failing) → HALF_OPEN (testing recovery).

All breakers use Redis for distributed state across pods (per AGENTS.md Rule 7).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import structlog
from redis.asyncio import Redis

from app.core.config import settings
from app.core.metrics import (
    circuit_breaker_open_total,
    circuit_breaker_state_changes_total,
)
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)


class AsyncRedisCircuitBreaker:
    """Async-native distributed circuit breaker using Redis.

    Replaces pybreaker + synchronous redis wrappers with a clean, fully async implementation.
    Shared across all pods via Redis counters and expirations.
    """

    def __init__(self, name: str):
        self.name = name
        self.fail_max = settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.reset_timeout = settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self._prefix = f"circuit_breaker:{name}"
        self._state_key = f"{self._prefix}:open"  # Exists with TTL when OPEN
        self._fails_key = f"{self._prefix}:fails"  # Counter for failures

    async def current_state(self, redis: Redis | None = None) -> str:
        """Get the current state from Redis."""
        try:
            if redis is None:
                redis = await get_redis()

            # If the open key exists, we are OPEN
            if await redis.exists(self._state_key):
                return "open"

            # If failures have reached fail_max but open key expired, we are trying to recover (HALF-OPEN)
            fails_val = await redis.get(self._fails_key)
            if fails_val is not None and int(fails_val) >= self.fail_max:
                return "half-open"

            return "closed"
        except Exception as exc:
            logger.warning("circuit_breaker.state_check_failed", api_name=self.name, error=str(exc))
            # Fail closed (allow traffic) if Redis goes down to maximize availability
            return "closed"

    async def record_success(self, redis: Redis) -> None:
        """Record a successful call, closing the circuit if it was HALF-OPEN."""
        try:
            # Atomic check-and-delete is technically safer, but deleting the fails key directly
            # resets the count. If we were half-open, we are now closed.
            fails_val = await redis.get(self._fails_key)
            if fails_val is not None:
                await redis.delete(self._fails_key)

                if int(fails_val) >= self.fail_max:
                    circuit_breaker_state_changes_total.labels(
                        api_name=self.name, from_state="half-open", to_state="closed"
                    ).inc()
                    logger.info("circuit_breaker.recovery_success", api_name=self.name)
        except Exception as exc:
            logger.warning("circuit_breaker.record_success_failed", api_name=self.name, error=str(exc))

    async def record_failure(self, redis: Redis) -> None:
        """Record a failure, tripping the circuit to OPEN if threshold reached."""
        try:
            fails = await redis.incr(self._fails_key)

            if fails == 1:
                # Set a TTL so intermittent errors don't slowly accumulate forever
                await redis.expire(self._fails_key, self.reset_timeout * 3)

            if fails >= self.fail_max:
                # Trip the circuit if not already open
                opened = await redis.set(self._state_key, "1", ex=self.reset_timeout, nx=True)
                if opened:
                    logger.warning("circuit_breaker.open", api_name=self.name)
                    circuit_breaker_state_changes_total.labels(
                        api_name=self.name, from_state="closed", to_state="open"
                    ).inc()
                    circuit_breaker_open_total.labels(api_name=self.name).inc()
        except Exception as exc:
            logger.warning("circuit_breaker.record_failure_failed", api_name=self.name, error=str(exc))

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute the protected function."""
        try:
            redis = await get_redis()

            state = await self.current_state(redis)
            if state == "open":
                logger.warning("circuit_breaker.open_rejected", api_name=self.name)
                return []

            try:
                # Can be either a native async function or a wrapped sync function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)

                await self.record_success(redis)
                return result

            except Exception as exc:
                await self.record_failure(redis)
                logger.error("circuit_breaker.function_error", api_name=self.name, error=str(exc))
                return []

        except Exception as exc:
            logger.error("circuit_breaker.fatal_error", api_name=self.name, error=str(exc))
            return []


def create_circuit_breaker(api_name: str) -> AsyncRedisCircuitBreaker:
    """Create a new AsyncRedisCircuitBreaker instance."""
    return AsyncRedisCircuitBreaker(api_name)


# === Global Singleton Circuit Breakers ===

perigon_breaker = create_circuit_breaker("perigon")
tavily_breaker = create_circuit_breaker("tavily")
exa_breaker = create_circuit_breaker("exa")
semantic_scholar_breaker = create_circuit_breaker("semantic_scholar")
openalex_breaker = create_circuit_breaker("openalex")
arxiv_breaker = create_circuit_breaker("arxiv")
pubmed_breaker = create_circuit_breaker("pubmed")
patentsview_breaker = create_circuit_breaker("patentsview")
lens_breaker = create_circuit_breaker("lens")


# === Safe Call Wrapper ===

async def call_with_circuit_breaker(
    breaker: AsyncRedisCircuitBreaker,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute function with circuit breaker protection.

    Backward compatible wrapper that proxies to breaker.call().
    """
    return await breaker.call(func, *args, **kwargs)

# Usage remains exactly the same:
#   from app.core.circuit_breaker import perigon_breaker, call_with_circuit_breaker
#   ...
#   return await call_with_circuit_breaker(perigon_breaker, async_func, *args)
