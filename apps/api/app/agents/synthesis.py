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
- Always cite your sources inline using markdown link format: [title](url)
  Example: According to [Fragrantica review](https://fragrantica.com/...), ...
  If no URL is available, fall back to plain text: [Source: title]
- NEVER infer, guess, or generate a URL — only use URLs that appear verbatim in the provided context.
  If the context has no URL for a source, write [title] without any link rather than inventing one.
- Deduplicate — if multiple specialists found the same fact, cite the best source once
- Rank information by relevance to the user's query
- Write in clear, professional prose — no bullet dumps unless the query calls for lists
- Be substantive: when sources provide rich detail (fragrance notes, ratings, descriptions,
  reviews, attributes, rankings), include that detail — do not reduce each item to one sentence
- Do NOT add dedicated "Evidence", "Confidence", "Gaps", or "Limitations" sections
- Do not hallucinate facts not present in the provided context
- If a specific gap would genuinely affect the user's decision, mention it in one sentence
  inline in natural prose — never as a standalone section
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
