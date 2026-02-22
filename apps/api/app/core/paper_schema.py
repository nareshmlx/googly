"""Paper schema normalization module.

Normalizes paper schemas from different research tools (Semantic Scholar, arXiv,
PubMed, Exa) into a consistent OpenAlex-compatible format expected by the formatter.

All public functions return list[dict] and never raise — defensive coding per
AGENTS.md Rule 4.
"""

from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


def _parse_year_from_date(date_str: str) -> int | None:
    """
    Extract year from ISO date string (e.g., '2024-01-15T10:30:00Z' -> 2024).

    Returns None if parsing fails — never raises.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Try common date formats
    date_formats = [
        "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 with Z
        "%Y-%m-%dT%H:%M:%S",  # ISO 8601 without Z
        "%Y-%m-%d",  # Date only
        "%Y/%m/%d",  # Alternative separator
        "%Y",  # Year only
    ]

    for fmt in date_formats:
        try:
            parsed = datetime.strptime(date_str.strip(), fmt)
            return parsed.year
        except (ValueError, AttributeError):
            continue

    # Fallback: try to extract first 4 digits as year
    try:
        # Extract digits and check if first 4 are a valid year
        digits = "".join(c for c in date_str if c.isdigit())
        if len(digits) >= 4:
            year = int(digits[:4])
            # Sanity check: year should be between 1900 and 2100
            if 1900 <= year <= 2100:
                return year
    except (ValueError, IndexError):
        pass

    return None


def normalize_paper_schema(papers: list[dict]) -> list[dict]:
    """
    Normalize paper schemas from different research tools into OpenAlex format.

    Handles field mapping across 4 research tools:
    - Semantic Scholar: year → publication_year, citation_count → cited_by_count,
                        bare paper_id hash → full https://www.semanticscholar.org/paper/{id}
    - arXiv: published (ISO date) → publication_year,
             arxiv_id → https://arxiv.org/abs/{id}
    - PubMed: published (date string) → publication_year,
              pmid / pre-built url → https://pubmed.ncbi.nlm.nih.gov/{pmid}/
    - Exa: published_date (ISO) → publication_year, url → paper_id,
           author (str) remapped to authors (list)

    The formatter (orchestrator.py:_format_openalex_paper_list_answer) expects:
    - publication_year (int or "n/a"): Display year
    - cited_by_count (int): Number of citations
    - doi (str): DOI link (primary — usually absent for non-academic Exa results)
    - paper_id (str): MUST be a full URL — used as clickable link when doi absent
    - title, content, authors, source (preserved as-is)

    paper_id is always normalised to a full URL (https://...) so the formatter
    can render it as a clickable link without additional processing.

    Missing fields are filled with sensible defaults:
    - publication_year: Parsed from date fields or None (shows "n/a" in output)
    - cited_by_count: 0 (not all sources provide citations)
    - paper_id: Full URL constructed from source-specific identifiers

    Returns normalized list[dict] on success, empty list on any failure.
    Never raises.
    """
    if not papers or not isinstance(papers, list):
        logger.warning("normalize_paper_schema.invalid_input", input_type=type(papers).__name__)
        return []

    normalized = []

    for idx, paper in enumerate(papers):
        if not isinstance(paper, dict):
            logger.warning(
                "normalize_paper_schema.invalid_paper",
                index=idx,
                paper_type=type(paper).__name__,
            )
            continue

        try:
            source = paper.get("source", "unknown")

            # Start with copy of original (preserve all fields)
            normalized_paper = paper.copy()

            # 1. Normalize publication_year field
            if "publication_year" not in normalized_paper:
                # Try different date fields based on source
                year = None

                # Direct year field (Semantic Scholar)
                if "year" in paper and paper["year"] is not None:
                    year_value = paper["year"]
                    if isinstance(year_value, int):
                        year = year_value
                    elif isinstance(year_value, str) and year_value.isdigit():
                        # Try to convert string to int
                        year = int(year_value)

                # Parse from ISO date fields (arXiv, Exa)
                if year is None and "published" in paper:
                    year = _parse_year_from_date(str(paper.get("published", "")))

                # Parse from published_date (Exa) — guard against None before str()
                if year is None and "published_date" in paper:
                    pd_val = paper.get("published_date")
                    if pd_val is not None:
                        year = _parse_year_from_date(str(pd_val))

                normalized_paper["publication_year"] = year

            # 2. Normalize cited_by_count field
            if "cited_by_count" not in normalized_paper:
                citation_count = 0

                # Semantic Scholar uses citation_count
                if "citation_count" in paper:
                    try:
                        citation_count = int(paper.get("citation_count", 0))
                    except (ValueError, TypeError):
                        citation_count = 0

                normalized_paper["cited_by_count"] = citation_count

            # 3. Normalize paper_id field into a clickable URL.
            #
            # The formatter reads `doi` then `paper_id` as the link — it never reads
            # the raw `url` field.  So paper_id MUST be a full URL, not a text
            # identifier like "arXiv:2401.12345" or "PMID:38012345".
            #
            # Priority:
            #   a) If paper_id is already a URL (starts with http) — keep it.
            #   b) arXiv  → build https://arxiv.org/abs/{arxiv_id}
            #   c) PubMed → use pre-built url field (already a full URL from the tool)
            #   d) Semantic Scholar with bare hash → build SS URL
            #   e) Any source with a url field → use it directly
            raw_paper_id = normalized_paper.get("paper_id", "") or ""
            if not str(raw_paper_id).startswith("http"):
                # paper_id is absent or a non-URL string — replace it with a real URL
                url_for_id = None

                if source == "arxiv" and paper.get("arxiv_id"):
                    url_for_id = f"https://arxiv.org/abs/{paper['arxiv_id']}"
                elif source == "pubmed" and paper.get("url"):
                    # papers_pubmed.py already builds the full PubMed URL
                    url_for_id = str(paper["url"])
                elif source == "pubmed" and paper.get("pmid"):
                    url_for_id = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                elif source == "semantic_scholar" and raw_paper_id:
                    # Bare hash like "abc123" — convert to canonical SS URL
                    url_for_id = f"https://www.semanticscholar.org/paper/{raw_paper_id}"
                elif source == "semantic_scholar" and paper.get("url"):
                    url_for_id = str(paper["url"])
                elif paper.get("url"):
                    # Exa and any other source — use the url field directly
                    url_for_id = str(paper["url"])

                if url_for_id:
                    normalized_paper["paper_id"] = url_for_id

            # 3b. Remap Exa's singular `author` field to `authors` list so all
            #     downstream consumers see a consistent field name.
            if (
                source == "exa"
                and "author" in normalized_paper
                and "authors" not in normalized_paper
            ):
                author_val = normalized_paper.get("author")
                normalized_paper["authors"] = [author_val] if author_val else []

            # 4. Ensure doi field exists (even if empty)
            if "doi" not in normalized_paper:
                normalized_paper["doi"] = ""

            normalized.append(normalized_paper)

        except Exception:
            logger.exception(
                "normalize_paper_schema.paper_error",
                index=idx,
                source=paper.get("source", "unknown"),
                title_preview=(str(paper.get("title", ""))[:60] if "title" in paper else None),
            )
            # Continue processing other papers on error
            continue

    logger.info(
        "normalize_paper_schema.complete",
        input_count=len(papers),
        output_count=len(normalized),
        skipped=len(papers) - len(normalized),
    )

    return normalized
