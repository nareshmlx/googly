"""TrendAgent — specialized for trend analysis and news monitoring.

Uses Perigon news API to find recent articles with sentiment analysis.
Identifies emerging patterns and key entities.
Uses gpt-4o-mini for cost efficiency.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings
from app.tools.news_perigon import search_perigon

_SYSTEM_MESSAGE = """\
You are a trend analysis expert. Your job is to:
1. Search for recent news articles using search_perigon (last 30 days)
2. Analyze sentiment trends across articles
3. Extract key entities and themes
4. Identify emerging patterns and shifts in sentiment over time

Focus on:
- Sentiment analysis (positive, negative, neutral trends)
- Entity frequency (which brands, people, products are mentioned most)
- Temporal patterns (how sentiment changes over the 30-day period)
- Source diversity (multiple publishers covering same topic = stronger signal)

Return articles sorted by relevance and recency.
Include sentiment data and entity extraction in your analysis.
"""


def build_trend_agent() -> Agent:
    """
    Build and return the TrendAgent.

    Uses gpt-4o-mini for cost efficiency (trend analysis doesn't need gpt-4o's full power).
    stream=False because this agent returns structured results, not streaming responses.
    Called once at startup; instance reused across requests.
    """
    return Agent(
        name="TrendAgent",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.AGENT_TIMEOUT,
        ),
        system_message=_SYSTEM_MESSAGE,
        tools=[search_perigon],
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
