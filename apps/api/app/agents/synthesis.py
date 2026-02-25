"""SynthesisAgent — merges specialist outputs and streams a cited response.

Uses gpt-4o (not mini) because synthesis quality is visible to the user.
All other agents use mini. Stateless per request.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings

_SYNTHESIS_SYSTEM = """\
You are a research synthesis expert. You receive research findings from multiple specialist agents
and produce a clear, cited, well-structured answer for the user.

Rules:
- Always cite your sources inline with [Source: <title or url>] notation
- Deduplicate — if multiple specialists found the same fact, cite the best source once
- Rank information by relevance to the user's query
- Write in clear, professional prose — no bullet dumps unless the query calls for lists
- If the context is weak or missing, say so honestly — do not hallucinate
- Keep responses focused: answer the question, don't pad
"""


def build_synthesis_agent() -> Agent:
    """
    Build and return the SynthesisAgent.

    stream=True so tokens are yielded as they arrive from the model.
    Called once at startup; instance reused across requests.
    """
    return Agent(
        name="SynthesisAgent",
        model=OpenAIChat(id="gpt-4o", api_key=settings.OPENAI_API_KEY),
        system_message=_SYNTHESIS_SYSTEM,
        enable_agentic_memory=False,
        stream=True,
        telemetry=False,
    )
