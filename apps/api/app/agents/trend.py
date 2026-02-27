"""TrendAgent — specialized for trend analysis and news monitoring.

Uses Perigon news API to find recent articles with sentiment analysis.
Identifies emerging patterns and key entities.
Uses gpt-4o-mini for cost efficiency.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings
from app.tools.news_perigon import search_perigon
from app.tools.social_reddit import search_reddit_posts
from app.tools.social_x import search_x_posts
from app.tools.social_youtube import search_youtube_videos

_SYSTEM_MESSAGE = """\
You are a trend analysis expert. Your job is to:
1. Search for recent news articles using search_perigon (last 30 days)
2. Pull social momentum signals using search_x_posts, search_reddit_posts, and search_youtube_videos
3. Analyze sentiment and engagement trends across both news and social data
4. Extract key entities and themes
5. Identify emerging patterns and shifts over time

Focus on:
- Sentiment analysis (positive, negative, neutral trends)
- Entity frequency (which brands, people, products are mentioned most)
- Temporal patterns (how sentiment changes over the 30-day period)
- Source diversity (news + multiple social platforms = stronger signal)

Return high-signal evidence sorted by relevance and recency.
Include sentiment, engagement indicators, and platform-specific observations.
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
        tools=[search_perigon, search_x_posts, search_reddit_posts, search_youtube_videos],
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
