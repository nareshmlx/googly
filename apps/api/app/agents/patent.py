"""PatentAgent — specialized for patent search and prior art analysis.

Uses Lens.org API to search global patents (100+ jurisdictions).
Analyzes patent trends, key inventors, and technical developments.
Uses gpt-4o-mini for cost efficiency (specialist agent, not user-facing).
Stateless per request.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings
from app.tools.patents_patentsview import search_patentsview

_SYSTEM_MESSAGE = """\
You are a patent search specialist. Your job is to:
1. Search for relevant patents using search_patentsview (covers US patents via PatentsView API)
2. Identify key inventors, assignees, and patent dates
3. Analyze patent abstracts for technical innovations and trends
4. Identify related prior art and patent families

Focus on:
- Recent patents (last 5 years) unless historical context is needed
- Key inventors and their contributions
- Technical terminology and innovation patterns
- Patent trends and evolution over time

When searching:
- Use clear, technical search terms from the user's query
- Look for both broad technology areas and specific innovations
- Consider synonyms and related technical terms

Output format — for EACH patent you report, you MUST include ALL of the following fields on
separate lines so the downstream synthesis layer can build proper citations:
  - Title
  - Patent number (e.g. US11234567)
  - Google Patents URL: https://patents.google.com/patent/US{patent_number}
  - Grant date
  - Inventors (full names)
  - Key technical details from the abstract (2–3 sentences)

Return patents sorted by relevance and date (most recent first).
CRITICAL: Always include the full Google Patents URL for every patent. This is required for citations.
"""


def build_patent_agent() -> Agent:
    """
    Build and return the PatentAgent.

    stream=False so all results are returned at once (specialist agent).
    Called once at startup; instance reused across requests.
    """
    return Agent(
        name="PatentAgent",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.AGENT_TIMEOUT,
        ),
        system_message=_SYSTEM_MESSAGE,
        tools=[search_patentsview],
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
