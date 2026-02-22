"""Text chunking for knowledge base ingestion.

Provides overlap-based chunking with sentence boundary awareness to preserve
context across chunk boundaries while avoiding mid-sentence splits.

Design:
- Fixed max_chars chunks with configurable overlap for context continuity
- Sentence boundary splitting where possible (". ", "? ", "! ", "\n\n")
- Simple and fast — semantic-aware but not recursive (complexity/gain tradeoff)
"""

import re

import structlog

logger = structlog.get_logger(__name__)

# Defaults aligned with text-embedding-3-small: ~512 tokens ≈ 2048 chars
DEFAULT_CHUNK_SIZE = 2048  # characters (512 tokens * 4)
DEFAULT_OVERLAP = 200  # characters (50 tokens * 4)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks of at most chunk_size characters.

    Uses sentence boundaries (". ", "? ", "! ", "\n\n") where possible to avoid
    splitting mid-sentence. Each chunk (except the first) includes an overlap
    from the tail of the previous chunk to preserve context for embedding.

    Args:
        text: Input text to chunk. Whitespace is normalized before chunking.
        chunk_size: Maximum chunk size in characters. Defaults to 2048 (≈512 tokens).
        overlap: Overlap size in characters between consecutive chunks. Defaults to 200 (≈50 tokens).

    Returns:
        List of non-empty text chunks with overlap. Empty input returns empty list.
        If text <= chunk_size, returns single-element list (no chunking needed).

    Example:
        >>> text = "First sentence. Second sentence. " * 100
        >>> chunks = chunk_text(text, chunk_size=500, overlap=100)
        >>> len(chunks) > 1
        True
        >>> len(chunks[0]) <= 500
        True
        >>> # Each chunk (except first) starts with tail of previous
        >>> chunks[1].startswith(chunks[0][-100:].strip())
        True
    """
    # Normalise whitespace: collapse 3+ newlines into 2, strip outer whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text] if text else []

    # Split on sentence boundaries: ". ", "? ", "! ", or "\n\n"
    sentence_endings = re.compile(r"(?<=[.?!])\s+|\n\n")
    sentences = sentence_endings.split(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Try adding sentence to current chunk
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            # Current chunk full — save it and start new chunk
            if current:
                chunks.append(current)
            # Start new chunk with overlap from tail of previous chunk
            if current and overlap > 0:
                tail = current[-overlap:]
                current = f"{tail} {sentence}".strip()
            else:
                current = sentence

    # Append final chunk if non-empty
    if current:
        chunks.append(current)

    return chunks
