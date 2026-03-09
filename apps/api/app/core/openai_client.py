"""Shared OpenAI client singleton for efficient connection pooling.

This module provides a single AsyncOpenAI client instance that is reused
across all API calls, avoiding the overhead of creating new httpx connection
pools per request.
"""

from functools import cache

from openai import AsyncOpenAI

from app.core.circuit import openai_breaker
from app.core.config import settings


# functools.cache creates a module-level singleton on first call — one
# connection pool per worker process, reused across all LLM calls.
@cache
def get_openai_client() -> AsyncOpenAI:
    """Return (or lazily create) the module-level AsyncOpenAI client."""
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def openai_completions_with_circuit_breaker(client: AsyncOpenAI):
    """Wrapper that applies circuit breaker to client.chat.completions.create."""
    original_create = client.chat.completions.create

    async def _wrapped_create(*args: object, **kwargs: object):
        return await openai_breaker.call_async(original_create, *args, **kwargs)

    return _wrapped_create
