"""ResearchAgent — specialized for academic/research queries.

Orchestrates 4 research tools in parallel (Semantic Scholar, arXiv, PubMed, Exa) and merges results.
Uses gpt-4o-mini for cost efficiency. Stateless per request.

All research tools are designed to never raise exceptions — they return [] on failure.
This agent can safely call all 4 tools in parallel without error handling overhead.
"""

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

from app.core.config import settings
from app.tools.papers_arxiv import search_arxiv
from app.tools.papers_pubmed import search_pubmed
from app.tools.papers_semantic_scholar import search_semantic_scholar
from app.tools.search_exa import search_exa

RESEARCH_SYSTEM_PROMPT = """\
You are a research assistant specialized in finding relevant academic papers and patents.

Your task: Given a research query, orchestrate the available research tools to find the most
relevant papers.

Available tools:
- search_semantic_scholar: 200M+ papers from all fields of science
- search_arxiv: 2M+ preprints in physics, mathematics, CS, and more
- search_pubmed: 35M+ biomedical citations from MEDLINE and life science journals
- search_exa: Neural/semantic search for supplemental web results

## CRITICAL: Output Format

You MUST return a JSON object with this exact structure:

{
  "papers": [
    {
      "title": "Paper title",
      "authors": ["Author 1", "Author 2"],
      "abstract": "Paper abstract text",
      "year": 2024,
      "url": "https://...",
      "source": "semantic_scholar" | "arxiv" | "pubmed" | "exa",
      "citations": 42,
      "relevance_score": 0.95
    }
  ],
  "total_count": 10,
  "sources_searched": ["semantic_scholar", "arxiv"]
}

## Instructions

1. Analyze the query to determine which tools to call
2. Call 2-4 relevant tools (call all 4 if query is broad, fewer if specific to a domain)
3. Merge and deduplicate results from multiple tools by title or DOI
4. Rank by relevance to the query (consider recency, citations, topic match)
5. Return top 10-15 most relevant papers in the JSON format above

IMPORTANT: Return ONLY the JSON object, no additional text or explanation.
"""


def build_research_agent() -> Agent:
    """
    Build and return the ResearchAgent.

    Uses gpt-4o-mini for cost efficiency (research orchestration is tool-heavy,
    not synthesis-heavy).
    stream=False since we need all results before merging/deduplication.
    System prompt enforces structured JSON output (Agno's OpenAIChat doesn't support
    response_format param).
    Called once at startup; instance reused across requests.
    """
    return Agent(
        name="ResearchAgent",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.AGENT_TIMEOUT,
        ),
        system_message=RESEARCH_SYSTEM_PROMPT,
        tools=[search_semantic_scholar, search_arxiv, search_pubmed, search_exa],
        enable_agentic_memory=False,
        stream=False,
        telemetry=False,
    )
