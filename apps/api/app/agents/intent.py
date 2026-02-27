"""IntentAgent — extracts structured query intent from user input.

Uses gpt-5-mini with JSON output mode. Runs first in the orchestrator pipeline
before any KB retrieval so specialists know what domain and entities to look for.
Stateless per request — no memory, no session state on the agent object itself.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings
from app.models.schemas import IntentResult  # noqa: F401 — exported for callers

_INTENT_SYSTEM = """\
You extract structured intent from a research query. Output ONLY valid JSON — no prose, no markdown.

Schema:
{
  "domain": "<primary research domain, e.g. fashion_materials, cosmetics, sustainability>",
  "query_type": "<one of: trend | research | patent | social | general>",
  "entities": ["<key entity 1>", "<key entity 2>"],
  "must_match_terms": ["<terms that must appear in relevant results>"],
  "expanded_terms": ["<optional related terms for recall>"],
  "domain_terms": ["<context terms that constrain domain relevance>"],
  "query_specificity": "<specific | broad>",
  "confidence": <float 0.0–1.0>,
  "is_research_query": <true | false>,
  "target_domain": "<domain like techcrunch.com if user explicitly names a specific website/publication, else null>"
}

Rules:
- domain: short snake_case label, as specific as possible
- query_type: best single match
  - "trend": fashion trends, beauty trends, market trends, trending topics, popularity analysis, sentiment analysis, news monitoring
    Examples: "What are the latest lipstick trends?", "Trending hair colors for spring 2024", "Most popular skincare routines"
  - "patent": patents, innovations, inventions, intellectual property, patent searches, prior art
    Examples: "Patents for anti-aging formulations", "Latest cosmetic delivery system patents", "Who holds the patent for retinol encapsulation?"
  - "research": academic papers, scientific studies, literature reviews, scholarly content
  - "social": social media content, influencer posts, UGC, platform-specific queries
  - "general": everything else not matching above categories
- entities: max 5, most important named things in the query
- must_match_terms: max 5; include exact user-mentioned specific entities when present
- expanded_terms: max 8 optional related terms to improve recall
- domain_terms: max 15 concise terms to constrain topical domain relevance
- query_specificity: "specific" when user mentions concrete named entities, else "broad"
- confidence: how certain you are about the domain classification
- is_research_query: true if the query is asking about research, academic papers, scientific studies, literature reviews, citations, or scholarly content; false otherwise
- target_domain: set to the domain (e.g. "techcrunch.com") if the query explicitly names a specific website or publication (e.g. "what does TechCrunch say", "according to Vogue", "Forbes article about", "from the BBC"); use domain format (e.g. "techcrunch.com" not "TechCrunch"); set to null if no specific site is mentioned
- Output ONLY the JSON object, nothing else
"""


def build_intent_agent() -> Agent:
    """
    Build and return the IntentAgent.

    Called once at app startup; the instance is reused across requests.
    Session state is not stored on the agent — each arun() call is independent.
    """
    return Agent(
        name="IntentAgent",
        model=OpenAIChat(id="gpt-5-mini", api_key=settings.OPENAI_API_KEY),
        system_message=_INTENT_SYSTEM,
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
