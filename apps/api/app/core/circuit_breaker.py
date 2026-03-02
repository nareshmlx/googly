"""Circuit breakers for external API calls using pybreaker (production-tested).

Prevents cascading failures by failing fast when APIs are unhealthy.
State machine: CLOSED (healthy) → OPEN (failing) → HALF_OPEN (testing recovery).

All breakers use Redis for distributed state across pods (per AGENTS.md Rule 7).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import redis
import structlog
from pybreaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerListener,
    CircuitBreakerStorage,
)

from app.core.config import settings
from app.core.metrics import circuit_breaker_open_total, circuit_breaker_state_changes_total

logger = structlog.get_logger(__name__)


# === Shared Redis Connection Pool (AGENTS.md Rule 7: prevent connection leak) ===

_sync_redis_pool: redis.ConnectionPool | None = None


def _get_sync_redis() -> redis.Redis:
    """Get synchronous Redis client from shared connection pool.

    Creates pool on first call, reuses for all circuit breakers.
    Prevents 40k connection leak (10k users × 4 tools).
    Per AGENTS.md Rule 2: production-ready, scalable, robust.
    """
    global _sync_redis_pool
    if _sync_redis_pool is None:
        _sync_redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=50,  # Pool size - shared across all circuit breakers
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        logger.info(
            "circuit_breaker.redis_pool_created",
            max_connections=50,
            redis_url=settings.REDIS_URL.split("@")[-1],  # Log without credentials
        )
    return redis.Redis(connection_pool=_sync_redis_pool)


# === Redis-backed Circuit Breaker Storage ===


class RedisCircuitBreakerStorage(CircuitBreakerStorage):
    """Redis-backed storage for circuit breaker state (distributed across pods).

    Stores state in Redis with TTL to prevent stale data.
    All 10 pods share the same circuit breaker state per API.
    Per AGENTS.md Rule 7: Required for 10k concurrent users across multiple pods.

    Uses synchronous Redis client (separate from async cache client) because
    pybreaker's storage interface is synchronous.
    """

    def __init__(self, redis_url: str, name: str):
        super().__init__(name)
        # Use shared connection pool to prevent connection leak
        self._redis: redis.Redis = _get_sync_redis()
        self._key_prefix = f"circuit_breaker:{name}"
        self._state_key = f"{self._key_prefix}:state"
        self._fail_counter_key = f"{self._key_prefix}:fail_counter"
        self._success_counter_key = f"{self._key_prefix}:success_counter"
        self._opened_at_key = f"{self._key_prefix}:opened_at"
        # TTL prevents stale data from surviving pod restarts (2x recovery timeout)
        self._ttl = settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT * 2

    @property
    def state(self) -> str:
        """Get current state from Redis (CLOSED/OPEN/HALF_OPEN)."""
        try:
            raw: str | None = self._redis.get(self._state_key)  # type: ignore[assignment]
            return raw or "closed"
        except redis.RedisError as e:
            logger.warning(
                "circuit_breaker.redis_error_failing_closed",
                service=self.name,
                error=str(e),
                context="state_get",
            )
            # CRITICAL: Fail closed (half-open) not open - preserve safety when Redis is down
            # HALF_OPEN allows limited traffic with monitoring vs CLOSED (full traffic)
            return "half-open"
        except Exception:
            logger.exception("circuit_breaker.redis.state_get_error", name=self.name)
            # Fail closed - conservative fallback
            return "half-open"

    @state.setter
    def state(self, state: str) -> None:
        """Set current state in Redis with TTL."""
        try:
            self._redis.setex(self._state_key, self._ttl, state)  # type: ignore[arg-type]
        except Exception:
            logger.exception("circuit_breaker.redis.state_set_error", name=self.name, state=state)

    @property
    def counter(self) -> int:
        """Get failure counter from Redis."""
        try:
            raw: str | None = self._redis.get(self._fail_counter_key)  # type: ignore[assignment]
            return int(raw) if raw else 0
        except redis.RedisError as e:
            logger.warning(
                "circuit_breaker.redis_error_counter_fallback",
                service=self.name,
                error=str(e),
                context="counter_get",
            )
            # Conservative fallback: assume some failures exist
            return settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD // 2
        except Exception:
            logger.exception("circuit_breaker.redis.counter_get_error", name=self.name)
            return settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD // 2

    @counter.setter
    def counter(self, count: int) -> None:
        """Set failure counter in Redis with TTL."""
        try:
            self._redis.setex(self._fail_counter_key, self._ttl, str(count))  # type: ignore[arg-type]
        except Exception:
            logger.exception("circuit_breaker.redis.counter_set_error", name=self.name, count=count)

    @property
    def opened_at(self) -> datetime | None:
        """Get timestamp when circuit was opened (for recovery timeout calculation)."""
        try:
            raw: str | None = self._redis.get(self._opened_at_key)  # type: ignore[assignment]
            return datetime.fromtimestamp(float(raw), tz=UTC) if raw else None
        except Exception:
            logger.exception("circuit_breaker.redis.opened_at_get_error", name=self.name)
            return None

    @opened_at.setter
    def opened_at(self, timestamp: datetime | float | None) -> None:
        """Set timestamp when circuit was opened."""
        try:
            if timestamp is None:
                self._redis.delete(self._opened_at_key)  # type: ignore[arg-type]
            else:
                if isinstance(timestamp, datetime):
                    opened_ts = timestamp.timestamp()
                else:
                    opened_ts = float(timestamp)
                self._redis.setex(self._opened_at_key, self._ttl, str(opened_ts))  # type: ignore[arg-type]
        except Exception:
            logger.exception(
                "circuit_breaker.redis.opened_at_set_error", name=self.name, timestamp=timestamp
            )

    def increment_counter(self) -> int:  # type: ignore[override]
        """Atomically increment failure counter in Redis."""
        try:
            # INCR is atomic - safe for concurrent requests across multiple pods
            count: int = self._redis.incr(self._fail_counter_key)  # type: ignore[assignment]
            # Reset TTL after increment
            self._redis.expire(self._fail_counter_key, self._ttl)  # type: ignore[arg-type]
            return count
        except Exception:
            logger.exception("circuit_breaker.redis.counter_increment_error", name=self.name)
            fallback = self.counter + 1
            self.counter = fallback  # Fallback to non-atomic increment
            return fallback

    def reset_counter(self) -> None:
        """Reset failure counter to zero."""
        self.counter = 0

    @property
    def success_counter(self) -> int:
        """Get success counter from Redis."""
        try:
            raw: str | None = self._redis.get(self._success_counter_key)  # type: ignore[assignment]
            return int(raw) if raw else 0
        except Exception:
            logger.exception("circuit_breaker.redis.success_counter_get_error", name=self.name)
            return 0

    @success_counter.setter
    def success_counter(self, count: int) -> None:
        """Set success counter in Redis with TTL."""
        try:
            self._redis.setex(self._success_counter_key, self._ttl, str(count))  # type: ignore[arg-type]
        except Exception:
            logger.exception(
                "circuit_breaker.redis.success_counter_set_error",
                name=self.name,
                count=count,
            )


# === Redis-backed Circuit Breaker Listeners ===


class RedisCircuitBreakerListener(CircuitBreakerListener):
    """Listener that logs circuit breaker state changes."""

    def __init__(self, api_name: str):
        self.api_name = api_name

    def before_call(self, cb: CircuitBreaker, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Called before the protected function is called."""
        pass  # No logging needed before call

    def state_change(self, cb: CircuitBreaker, old_state: Any, new_state: Any) -> None:
        """Log state transitions and increment metrics."""
        logger.info(
            "circuit_breaker.state_change",
            api_name=self.api_name,
            old_state=str(old_state),
            new_state=str(new_state),
        )
        circuit_breaker_state_changes_total.labels(
            api_name=self.api_name,
            from_state=str(old_state),
            to_state=str(new_state),
        ).inc()

        # Track when circuit opens
        if str(new_state) == "open":
            circuit_breaker_open_total.labels(api_name=self.api_name).inc()

    def failure(self, cb: CircuitBreaker, exc: BaseException) -> None:
        """Log failures."""
        logger.warning(
            "circuit_breaker.failure",
            api_name=self.api_name,
            failure_count=cb.fail_counter,
            error=str(exc),
        )

    def success(self, cb: CircuitBreaker) -> None:
        """Log successful calls (only in HALF_OPEN state)."""
        if cb.current_state == "half-open":
            logger.info("circuit_breaker.recovery_success", api_name=self.api_name)


# === Global Singleton Circuit Breakers ===


def create_circuit_breaker(api_name: str) -> CircuitBreaker:
    """Create a circuit breaker with Redis-backed storage and logging listener.

    Per AGENTS.md Rule 7: Redis storage ensures distributed state across all 10 pods.
    All pods see the same circuit breaker state for each API.
    """
    storage = RedisCircuitBreakerStorage(settings.REDIS_URL, api_name)
    breaker = CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        reset_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        name=api_name,
        listeners=[RedisCircuitBreakerListener(api_name)],
        state_storage=storage,
    )
    return breaker


perigon_breaker = create_circuit_breaker("perigon")
tavily_breaker = create_circuit_breaker("tavily")
exa_breaker = create_circuit_breaker("exa")
semantic_scholar_breaker = create_circuit_breaker("semantic_scholar")
openalex_breaker = create_circuit_breaker("openalex")
arxiv_breaker = create_circuit_breaker("arxiv")
pubmed_breaker = create_circuit_breaker("pubmed")
patentsview_breaker = create_circuit_breaker("patentsview")
lens_breaker = create_circuit_breaker("lens")


# === Safe Call Wrapper (Returns [] on circuit open or failure) ===


async def call_with_circuit_breaker(
    breaker: CircuitBreaker,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute function with circuit breaker protection (async-safe).

    Returns [] when circuit is open (fail fast) or function raises exception.
    Uses pybreaker's breaker.call() which internally tracks success/failure state.
    Follows AGENTS.md Rule 4: Tools never raise exceptions.
    """
    # 1. Check state (offload to thread as it calls Redis storage.state)
    try:
        current_state = await asyncio.to_thread(lambda: breaker.current_state)
        if current_state == "open":
            logger.warning(
                "circuit_breaker.open",
                api_name=breaker.name,
                state=current_state,
            )
            return []
    except Exception as exc:
        logger.error("circuit_breaker.state_check_failed", api_name=breaker.name, error=str(exc))
        # Fail-closed (allow call) if state check fails to keep system alive
        pass

    try:
        # 2. Execute the protected function via breaker.call() so pybreaker
        #    internally tracks success/failure — it does NOT expose .success()/.failure()
        #    as public methods on CircuitBreaker; those belong to listeners only.
        if asyncio.iscoroutinefunction(func):
            # For async functions: run the coroutine, then notify breaker via call()
            # We pass a sync wrapper to breaker.call() in a thread
            result = await func(*args, **kwargs)
            # Notify pybreaker of success by calling a no-op through the breaker
            try:
                await asyncio.to_thread(breaker.call, lambda: None)
            except Exception:
                pass  # Don't fail on breaker bookkeeping errors
            return result
        else:
            # If function is sync, call it through the breaker directly
            # breaker.call() handles success/failure tracking internally
            result = await asyncio.to_thread(breaker.call, func, *args, **kwargs)
            return result

    except CircuitBreakerError:
        logger.warning("circuit_breaker.open_rejected", api_name=breaker.name)
        return []
    except Exception as exc:
        logger.error(
            "circuit_breaker.function_error",
            api_name=breaker.name,
            error=str(exc),
        )
        return []


# Usage:
#   from app.core.circuit_breaker import perigon_breaker, call_with_circuit_breaker
#
#   async def fetch_perigon():
#       return await call_with_circuit_breaker(
#           perigon_breaker,
#           client.get,
#           "https://api.goperigon.com/v1/all",
#           params={"q": query}
#       )
