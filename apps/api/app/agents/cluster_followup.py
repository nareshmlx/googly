"""Cluster follow-up agent factory."""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings

_SYSTEM_PROMPT = """\
You answer follow-up questions about one insight cluster using only provided context.

Rules:
- Use only facts in the provided context snippets.
- If context is insufficient, clearly say what is missing.
- Cite source titles and URLs as markdown links when URL is available.
- Do not use external web knowledge.
"""


def build_cluster_followup_agent() -> Agent:
    """Build a stateless streaming follow-up agent."""
    return Agent(
        name="ClusterFollowupAgent",
        model=OpenAIChat(id="gpt-4o", api_key=settings.OPENAI_API_KEY),
        system_message=_SYSTEM_PROMPT,
        enable_agentic_memory=False,
        stream=True,
        telemetry=False,
    )
