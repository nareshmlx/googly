"""
OpenAlex Search Tool for Agno Agent

This tool enables searching OpenAlex for research papers.
"""

import httpx
from typing import List, Dict, Any, Optional

# Global storage for discovered papers (can be accessed by Streamlit if needed)
_discovered_papers = []

def get_discovered_papers():
    """Return the list of discovered papers."""
    return _discovered_papers

def clear_discovered_papers():
    """Clear the discovered papers list."""
    global _discovered_papers
    _discovered_papers = []

def search_openalex(query: str) -> str:
    """
    Search OpenAlex for research papers related to a specific query.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing search results with paper information
    """
    global _discovered_papers
    
    email = "ridajavedcmd@gmail.com"  # Using the email from the adapter
    base_url = "https://api.openalex.org/works"
    
    params = {
        "search": query,
        "per-page": 5,
        "mailto": email,
        "sort": "cited_by_count:desc"
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
        results = data.get("results", [])
        
        if not results:
            return f"No research papers found for query: {query}"
            
        formatted_results = [f"📚 OpenAlex Search Results for '{query}'\n"]
        formatted_results.append(f"Found {data.get('meta', {}).get('count', 0)} papers (showing top 5)\n")
        
        for i, work in enumerate(results, 1):
            # Extract info
            title = work.get("title", "Untitled")
            publication_year = work.get("publication_year", "N/A")
            cited_by_count = work.get("cited_by_count", 0)
            
            # Authors
            authorships = work.get("authorships", [])
            authors = []
            for authorship in authorships:
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])
            author_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
            
            # Abstract (reconstruct)
            abstract = ""
            inverted_index = work.get("abstract_inverted_index")
            if inverted_index:
                word_positions = []
                for word, positions in inverted_index.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join([word for _, word in word_positions])
                if len(abstract) > 200:
                    abstract = abstract[:200] + "..."
            
            # PDF URL
            pdf_url = work.get("open_access", {}).get("is_oa", False) and work.get("best_oa_location", {}).get("pdf_url")
            
            # Store for discovery
            paper_data = {
                "id": work.get("id"),
                "title": title,
                "authors": authors,
                "year": publication_year,
                "cited_by_count": cited_by_count,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "doi": work.get("doi"),
                "landing_page_url": work.get("id") # OpenAlex ID is a URL
            }
            
            if not any(p['id'] == paper_data['id'] for p in _discovered_papers):
                _discovered_papers.append(paper_data)

            # Format string output
            formatted_results.append(f"\n📄 Paper {i}: {title}")
            formatted_results.append(f"  👥 Authors: {author_str}")
            formatted_results.append(f"  📅 Year: {publication_year} | 📢 Citations: {cited_by_count}")
            if abstract:
                formatted_results.append(f"  📝 Abstract: {abstract}")
            if pdf_url:
                formatted_results.append(f"  🔗 PDF: {pdf_url}")
            formatted_results.append(f"  🌐 Link: {work.get('id')}")
            
        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching OpenAlex: {str(e)}"
