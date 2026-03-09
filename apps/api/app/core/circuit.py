"""Circuit breaker patterns for resilient external API calls.

Uses aiobreaker to prevent cascading failures when upstream services are down.
Can be combined with tenacity retries (circuit breaker should usually be the
outer layer to immediately fail once the threshold is reached).
"""

from typing import Any, Callable, TypeVar

import structlog
from aiobreaker import CircuitBreaker, CircuitBreakerListener

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class LogListener(CircuitBreakerListener):
    """Logs circuit breaker state transitions."""

    def __init__(self, name: str):
        self.name = name

    def state_change(self, breaker: CircuitBreaker, old_state: Any, new_state: Any) -> None:
        logger.warning(
            "circuit_breaker.state_change",
            name=self.name,
            old_state=str(old_state),
            new_state=str(new_state),
        )

    def failure(self, breaker: CircuitBreaker, exc: Exception) -> None:
        logger.debug("circuit_breaker.failure", name=self.name, error=str(exc))

    def success(self, breaker: CircuitBreaker) -> None:
        pass


def create_breaker(
    name: str,
    fail_max: int = 5,
    timeout_seconds: int = 60,
    **kwargs: Any,
) -> CircuitBreaker:
    """Create a configured aiobreaker instance.

    Args:
        name: Identifier for logs/metrics
        fail_max: Consecutive failures before opening the circuit
        timeout_seconds: Seconds to wait in 'open' state before trying 'half-open'
    """
    breaker = CircuitBreaker(fail_max=fail_max, timeout_duration=timeout_seconds, **kwargs)
    breaker.add_listeners(LogListener(name))
    return breaker


# Pre-defined breakers for core external dependencies
openai_breaker = create_breaker("openai", fail_max=10, timeout_seconds=30)
ensemble_breaker = create_breaker("ensembledata", fail_max=5, timeout_seconds=60)
arxiv_breaker = create_breaker("arxiv", fail_max=5, timeout_seconds=120)
openalex_breaker = create_breaker("openalex", fail_max=5, timeout_seconds=60)
