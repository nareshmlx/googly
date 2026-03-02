"""LLM service wrappers for direct OpenAI calls."""

import structlog
from app.core.openai_client import get_openai_client

logger = structlog.get_logger(__name__)


async def chat_completion(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Simple wrapper for OpenAI chat completions, returning the message content."""
    client = get_openai_client()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (response.choices[0].message.content or "").strip()
        return content
    except Exception as exc:
        logger.error("llm.chat_completion_failed", error=str(exc))
        raise
