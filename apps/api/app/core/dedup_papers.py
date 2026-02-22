"""Paper deduplication module.

Deduplicates research papers across multiple tools using:
1. Exact DOI matching (case-insensitive)
2. Fuzzy title matching (SequenceMatcher with 85% threshold)

Priority order: Semantic Scholar > arXiv > PubMed > Exa
(Semantic Scholar has the most complete metadata and citations)

All functions are defensive and handle missing/malformed fields gracefully.
"""

import re
from difflib import SequenceMatcher

import structlog

logger = structlog.get_logger(__name__)

# Priority order for source selection (when same paper found in multiple sources)
_SOURCE_PRIORITY = {
    "semantic_scholar": 1,
    "arxiv": 2,
    "pubmed": 3,
    "exa": 4,
}


def normalize_title(title: str) -> str:
    """
    Normalize paper title for fuzzy matching.

    Removes punctuation, lowercases, collapses whitespace.
    Returns empty string if input is invalid.
    """
    if not title or not isinstance(title, str):
        return ""

    # Lowercase
    normalized = title.lower()

    # Remove punctuation (keep alphanumeric and spaces)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    # Collapse multiple spaces to single space
    normalized = re.sub(r"\s+", " ", normalized)

    # Strip leading/trailing whitespace
    return normalized.strip()


def titles_match(title1: str, title2: str, threshold: float = 0.85) -> bool:
    """
    Check if two titles are similar using fuzzy matching.

    Uses SequenceMatcher ratio with 85% default threshold.
    Returns False if either title is invalid/empty.
    """
    if not title1 or not title2 or not isinstance(title1, str) or not isinstance(title2, str):
        return False

    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    # If either normalized title is empty, they don't match
    if not norm1 or not norm2:
        return False

    # Quick check: if lengths differ by more than 50%, likely not a match
    len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
    if len_ratio < 0.5:
        return False

    # Use SequenceMatcher for fuzzy comparison
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


def _get_source_priority(paper: dict) -> int:
    """Get priority rank for a paper's source (lower is better)."""
    source = str(paper.get("source", "")).lower()
    return _SOURCE_PRIORITY.get(source, 999)


def _normalize_doi(doi: str | None) -> str:
    """Normalize DOI for comparison (lowercase, strip whitespace)."""
    if not doi or not isinstance(doi, str):
        return ""
    return doi.strip().lower()


def deduplicate_papers(
    semantic_scholar: list[dict] | None = None,
    arxiv: list[dict] | None = None,
    pubmed: list[dict] | None = None,
    exa: list[dict] | None = None,
) -> list[dict]:
    """
    Deduplicate papers across multiple research tools.

    Deduplication logic:
    1. First pass: exact DOI matching (case-insensitive)
    2. Second pass: fuzzy title matching (85% threshold)
    3. Priority order: Semantic Scholar > arXiv > PubMed > Exa

    Args:
        semantic_scholar: Papers from Semantic Scholar API
        arxiv: Papers from arXiv API
        pubmed: Papers from PubMed API
        exa: Papers from Exa search

    Returns:
        Deduplicated list of papers with source attribution.
        Empty list if all inputs are None or empty.
    """
    # Combine all papers with defensive null handling
    all_papers: list[dict] = []
    for source_list in [semantic_scholar, arxiv, pubmed, exa]:
        if source_list and isinstance(source_list, list):
            all_papers.extend(source_list)

    if not all_papers:
        logger.info("dedup_papers.no_papers")
        return []

    raw_count = len(all_papers)

    # Track seen papers by DOI and title
    seen_dois: set[str] = set()
    seen_titles: list[str] = []  # Store normalized titles for fuzzy matching
    deduped_papers: list[dict] = []

    for paper in all_papers:
        if not isinstance(paper, dict):
            continue

        # Get paper metadata
        doi = _normalize_doi(paper.get("doi"))
        title = str(paper.get("title", "")).strip()

        # Skip papers with no title
        if not title:
            continue

        # First pass: exact DOI deduplication
        if doi and doi in seen_dois:
            continue

        # Second pass: fuzzy title matching
        normalized_title = normalize_title(title)
        if not normalized_title:
            continue

        # Check if this title matches any previously seen title
        is_duplicate = False
        for seen_title in seen_titles:
            if titles_match(normalized_title, seen_title):
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Not a duplicate - add to results
        deduped_papers.append(paper)

        # Track this paper to prevent future duplicates
        if doi:
            seen_dois.add(doi)
        seen_titles.append(normalized_title)

    # Sort by source priority (Semantic Scholar first, then arXiv, etc.)
    deduped_papers.sort(key=_get_source_priority)

    # Calculate deduplication stats
    deduped_count = len(deduped_papers)
    duplicates_removed = raw_count - deduped_count
    duplicate_rate = (duplicates_removed / raw_count) if raw_count > 0 else 0.0

    logger.info(
        "dedup_papers.complete",
        raw_count=raw_count,
        deduped_count=deduped_count,
        duplicates_removed=duplicates_removed,
        duplicate_rate=round(duplicate_rate, 3),
    )

    return deduped_papers
