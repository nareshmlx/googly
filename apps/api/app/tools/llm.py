"""LLM tool wrappers for provider calls."""

from __future__ import annotations

import structlog
from openai.types.chat import ChatCompletionMessageParam

from app.core.openai_client import get_openai_client

logger = structlog.get_logger(__name__)


async def chat_completion(
    messages: list[ChatCompletionMessageParam],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Simple wrapper for OpenAI chat completions, returning message content."""
    client = get_openai_client()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("llm.chat_completion_failed", error=str(exc))
        raise


async def expand_project_description(description: str) -> str:
    """Expand short project text into richer intent extraction context."""
    client = get_openai_client()
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Expand short project descriptions into 2-3 rich sentences "
                        "for research intent extraction. Include domain context, "
                        "core topics, and likely information needs. Output only the "
                        "expanded description."
                    ),
                },
                {"role": "user", "content": description},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(
            "llm.expand_project_description_failed",
            description_preview=description[:80],
            error=str(exc),
        )
        return ""
