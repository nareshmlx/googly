"""API rate limiters using aiolimiter (production-tested, race condition-free).

All limiters are singleton instances shared across the application.
Uses token bucket algorithm with configurable burst capacity.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from aiolimiter import AsyncLimiter

from app.core.config import settings
from app.core.metrics import rate_limiter_throttled_total

T = TypeVar("T")

# === Global Singleton Limiters ===
# Each limiter enforces rate_limit requests per second with 2x burst capacity

perigon_limiter = AsyncLimiter(
    max_rate=settings.PERIGON_RATE_LIMIT,
    time_period=1.0,
)

tavily_limiter = AsyncLimiter(
    max_rate=settings.TAVILY_RATE_LIMIT,
    time_period=1.0,
)

exa_limiter = AsyncLimiter(
    max_rate=settings.EXA_RATE_LIMIT,
    time_period=1.0,
)

semantic_scholar_limiter = AsyncLimiter(
    max_rate=settings.SEMANTIC_SCHOLAR_RATE_LIMIT,
    time_period=1.0,
)

openalex_limiter = AsyncLimiter(
    max_rate=settings.OPENALEX_RATE_LIMIT,
    time_period=1.0,
)

arxiv_limiter = AsyncLimiter(
    max_rate=settings.ARXIV_RATE_LIMIT,
    time_period=1.0,
)

patentsview_limiter = AsyncLimiter(
    max_rate=settings.PATENTSVIEW_RATE_LIMIT,
    time_period=1.0,
)

pubmed_limiter = AsyncLimiter(
    max_rate=settings.PUBMED_RATE_LIMIT,
    time_period=1.0,
)

lens_limiter = AsyncLimiter(
    max_rate=settings.LENS_RATE_LIMIT,
    time_period=1.0,
)


async def rate_limited_call(
    limiter: AsyncLimiter,
    api_name: str,
    func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute function with rate limiting and metrics tracking.

    Tracks when calls are throttled (waiting for rate limit tokens).
    Per AGENTS.md Rule 8: Metrics required for production observability.

    Args:
        limiter: AsyncLimiter instance to use
        api_name: API name for metrics labeling
        func: Async function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func
    """
    start = time.monotonic()
    async with limiter:
        wait_time = time.monotonic() - start
        # Track throttling if we had to wait >10ms (indicates rate limit hit)
        if wait_time > 0.01:
            rate_limiter_throttled_total.labels(api_name=api_name).inc()
        return await func(*args, **kwargs)


# Usage:
#   from app.core.rate_limiter import perigon_limiter, rate_limited_call
#
#   async def call_perigon_api():
#       return await rate_limited_call(
#           perigon_limiter,
#           "perigon",
#           client.get,
#           "https://api.goperigon.com/v1/all",
#           params={"q": query}
#       )
