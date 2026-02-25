"""SearchFallbackAgent — runs when KB score < 0.70.

Orchestrates 3 search APIs in parallel (Tavily, Exa, Perigon) and merges results.
Uses gpt-4o-mini for cost efficiency. Stateless per request.

This agent is invoked when the KB retriever returns a score below the threshold,
signalling that the knowledge base does not contain sufficient information to
answer the user's query. In this case, the agent orchestrates parallel searches
across multiple external APIs and returns unified, deduplicated results.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings
from app.tools.news_perigon import search_perigon
from app.tools.search_exa import search_exa
from app.tools.search_tavily import search_tavily

_SEARCH_FALLBACK_SYSTEM = """\
You are a search orchestration specialist. Your job is to gather comprehensive information
from multiple search APIs when the knowledge base does not have sufficient data.

Your task:
1. Call ALL three search tools in parallel: search_tavily, search_exa, search_perigon
2. Wait for all results to complete (some may return empty results — that's fine)
3. Merge all results into a single comprehensive list
4. Deduplicate by URL (if the same URL appears in multiple sources, keep only one)
5. Return the unified results

Always call all three tools, even if one or two return empty results.
The goal is to cast a wide net across multiple search providers for maximum coverage.
"""


def build_search_fallback_agent() -> Agent:
    """
    Build and return the SearchFallbackAgent.

    stream=False so all results are returned at once (not streaming tokens).
    This agent orchestrates tool calls and returns structured data, not narrative text.
    Called once at startup; instance reused across requests.

    Uses gpt-4o-mini for cost efficiency — this is tool orchestration, not synthesis.
    """
    return Agent(
        name="SearchFallbackAgent",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.AGENT_TIMEOUT,
        ),
        system_message=_SEARCH_FALLBACK_SYSTEM,
        tools=[search_tavily, search_exa, search_perigon],
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
