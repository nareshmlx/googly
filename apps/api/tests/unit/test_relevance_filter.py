"""Unit tests for ingest relevance filter helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tasks.ingest_project import (
    _filter_relevance,
    _filter_stage1_embedding,
    _filter_stage2_llm,
    _query_for_papers,
    _query_for_patents,
    _query_for_social,
)


@pytest.mark.asyncio
async def test_stage1_drops_bottom_40_percent() -> None:
    """Stage1 keeps items at or above the 40th percentile similarity."""
    items = [{"title": f"Item {i}", "abstract": f"Content {i}"} for i in range(10)]

    intent_vec = [1.0, 0.0, 0.0]
    item_vecs = [[1.0, float(i), 0.0] for i in range(10)]

    with patch("app.tasks.ingest_project.embed_texts", new=AsyncMock()) as embed_texts_mock:
        embed_texts_mock.side_effect = [[intent_vec], item_vecs]
        result = await _filter_stage1_embedding(items=items, intent_text="intent", redis=None)

    assert len(result) == 6


@pytest.mark.asyncio
async def test_filter_relevance_skips_social() -> None:
    """Top-level filter bypasses both stages for social sources."""
    items = [{"title": "Video", "content": "Beauty trend"}]

    result = await _filter_relevance(
        items=items,
        intent_text="beauty",
        source="social_tiktok",
        redis=None,
    )

    assert result == items


@pytest.mark.asyncio
async def test_stage2_drops_irrelevant_by_llm_response() -> None:
    """Stage2 keeps only items marked relevant=1 by the LLM response."""
    items = [
        {"title": "Fragrance Chemistry", "abstract": "Scent molecules."},
        {"title": "Car Engine Design", "abstract": "Combustion thermodynamics."},
    ]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"items": [{"id": 0, "relevant": 1}, {"id": 1, "relevant": 0}]}'
            )
        )
    ]

    with patch("app.tasks.ingest_project.AsyncOpenAI") as openai_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        openai_cls.return_value = mock_client

        result = await _filter_stage2_llm(
            items=items, intent_text="luxury fragrance", source="paper"
        )

    assert len(result) == 1
    assert result[0]["title"] == "Fragrance Chemistry"


@pytest.mark.asyncio
async def test_filter_relevance_paper_min_survivors_when_stage2_is_too_strict() -> None:
    """Paper filtering keeps a minimum number of candidates if stage2 returns too few."""
    items = [{"title": f"Paper {i}", "abstract": "Relevant abstract"} for i in range(8)]

    with (
        patch(
            "app.tasks.ingest_project._filter_stage1_embedding",
            new=AsyncMock(return_value=items),
        ),
        patch(
            "app.tasks.ingest_project._filter_stage2_llm",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await _filter_relevance(
            items=items,
            intent_text="beauty science",
            source="paper",
            redis=None,
        )

    assert len(result) == 6
    assert [item["title"] for item in result] == [f"Paper {i}" for i in range(6)]


def test_query_for_patents_falls_back_to_keywords_and_entities() -> None:
    """Patent query helper falls back when explicit patent filter is missing."""
    intent = {
        "search_filters": {},
        "entities": ["retinol"],
        "keywords": ["encapsulation", "stability", "delivery system"],
    }

    query = _query_for_patents(intent)

    assert query == "retinol encapsulation stability delivery system"


def test_query_for_papers_falls_back_to_project_context() -> None:
    """Paper query uses project text when intent filters are empty."""
    query = _query_for_papers(
        {"search_filters": {}, "entities": [], "keywords": []},
        social_filter="",
        project_title="Hydrating sunscreen",
        project_description="UV filters with skin barrier support for sensitive skin.",
    )

    assert query.startswith("Hydrating sunscreen")


def test_query_for_social_uses_keywords_before_project_context() -> None:
    """Social query derives from intent keywords when social filter is absent."""
    query = _query_for_social(
        {
            "search_filters": {},
            "keywords": ["glass skin", "hydrating essence"],
            "entities": [],
        },
        project_title="Ignored title",
        project_description="Ignored description",
    )

    assert query == "glass skin hydrating essence"
