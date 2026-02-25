"""QueryEnhancerAgent — normalize user query while preserving user-specified terms.

This runs before intent extraction to reduce noisy phrasing and improve
downstream tool query construction. The output is strict JSON.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings

_SYSTEM_MESSAGE = """\
You normalize research queries for retrieval systems.
Output ONLY valid JSON. Do not output prose.

Schema:
{
  "rewritten_query": "<cleaned query that preserves user's specific entities>",
  "must_match_terms": ["<specific user-mentioned term 1>", "<term 2>"],
  "expanded_terms": ["<optional related terms for recall>"],
  "domain_terms": ["<domain grounding terms that improve topical precision>"],
  "query_specificity": "<specific|broad>"
}

Rules:
- Preserve exact user-mentioned entities (ingredients, compounds, product names, brands).
- Remove filler language only ("latest papers on", "tell me about", etc.).
- Never replace specific entities with generic abstractions.
- domain_terms should be 1-4 short terms grounded in the actual query intent.
- expanded_terms should be 0-8 optional recall terms.
- query_specificity = "specific" when user mentions concrete entities or names.
- Return JSON only.
"""


def build_query_enhancer_agent() -> Agent:
    """Build and return the pre-intent query enhancer agent."""
    return Agent(
        name="QueryEnhancerAgent",
        model=OpenAIChat(
            id="gpt-5-mini",
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.AGENT_TIMEOUT,
        ),
        system_message=_SYSTEM_MESSAGE,
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
