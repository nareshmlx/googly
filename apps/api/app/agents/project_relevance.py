"""ProjectRelevanceAgent — decides which of a user's projects are relevant to a query.

Uses gpt-5-mini with JSON output. Returns a list of project_ids to search.
Falls back to the primary project_id if parsing fails — never returns an empty list,
because an empty list would produce a response with no KB context at all.
"""


import structlog
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings

logger = structlog.get_logger(__name__)

_RELEVANCE_SYSTEM = """\
You decide which of a user's research projects are relevant to answer a query.

You will receive:
- query: the user's question
- projects: list of {id, title, description} for all their projects

Output ONLY valid JSON:
{"relevant_project_ids": ["<uuid>", ...]}

Rules:
- Include a project if its subject matter overlaps with the query, even partially
- Always include at least one project — the most relevant one if none are obvious
- Maximum 3 projects — more context is not always better
- Output ONLY the JSON object, nothing else
"""


def build_project_relevance_agent() -> Agent:
    """
    Build the ProjectRelevanceAgent.

    Called once at startup and reused across requests. Stateless per arun() call.
    """
    return Agent(
        name="ProjectRelevanceAgent",
        model=OpenAIChat(id="gpt-5-mini", api_key=settings.OPENAI_API_KEY),
        system_message=_RELEVANCE_SYSTEM,
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
