"""Shared OpenAI client singleton for efficient connection pooling.

This module provides a single AsyncOpenAI client instance that is reused
across all API calls, avoiding the overhead of creating new httpx connection
pools per request.
"""

from openai import AsyncOpenAI

from app.core.config import settings

# Module-level singleton — one connection pool per worker process, reused across
# all LLM calls. Creating a new AsyncOpenAI() per call creates a new
# httpx.AsyncClient on every invocation, which wastes connections under load.
_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return (or lazily create) the module-level AsyncOpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client
