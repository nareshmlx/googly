"""Unit tests for ingest relevance filter helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tasks.ingest_project import (
    _filter_relevance,
    _filter_stage1_embedding,
    _filter_stage2_llm,
    _project_anchor_terms,
    _query_for_papers,
    _query_for_patents,
    _query_for_social,
    _social_must_terms,
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
async def test_filter_relevance_applies_strict_social_gate() -> None:
    """Social sources require lexical must-match + similarity threshold in strict mode."""
    items = [
        {"title": "Retinol review", "content": "Retinol night routine for acne marks"},
        {"title": "Makeup hacks", "content": "Lipstick hacks and contour routine"},
    ]
    intent_text = '{"must_match_terms":["retinol"],"query_specificity":"specific"}'
    intent_vec = [1.0, 0.0, 0.0]
    item_vecs = [
        [0.95, 0.05, 0.0],
    ]

    with patch("app.tasks.ingest_project.embed_texts", new=AsyncMock()) as embed_texts_mock:
        embed_texts_mock.side_effect = [[intent_vec], item_vecs]
        result = await _filter_relevance(
            items=items,
            intent_text=intent_text,
            source="social_tiktok",
            redis=None,
            must_match_terms=["retinol"],
        )

    assert len(result) == 1
    assert result[0]["title"] == "Retinol review"


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

    with patch("app.tasks.ingest_project.get_openai_client") as get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        get_client.return_value = mock_client

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


def test_query_for_social_keeps_specific_terms_beyond_first_four_keywords() -> None:
    """Social query should not drop later specific terms like retinol."""
    query = _query_for_social(
        {
            "search_filters": {},
            "must_match_terms": ["retinol"],
            "keywords": [
                "skincare",
                "barrier",
                "sensitive",
                "night",
                "retinol",
                "encapsulation",
            ],
            "entities": [],
            "domain_terms": ["cosmetics"],
        },
        project_title="Retinol project",
        project_description="Retinol stabilization",
    )

    lowered = query.lower()
    assert "retinol" in lowered
    assert "encapsulation" in lowered


def test_social_must_terms_uses_specific_non_broad_keywords_when_explicit_terms_missing() -> None:
    """Specific social queries should derive strict terms from entities/keywords."""
    terms = _social_must_terms(
        {
            "query_specificity": "specific",
            "must_match_terms": [],
            "entities": ["retinol"],
            "keywords": ["skincare", "encapsulation", "beauty"],
        },
        query_terms=["beauty", "retinol", "encapsulation"],
    )
    assert "retinol" in terms
    assert "encapsulation" in terms
    assert "beauty" not in terms


def test_project_anchor_terms_prioritizes_project_description_specific_terms() -> None:
    """Anchor extraction should include project-specific ingredient/process terms."""
    anchors = _project_anchor_terms(
        {
            "keywords": ["retinol", "encapsulation"],
            "entities": ["retinoid"],
        },
        social_filter="#retinol #skincare",
        project_title="Retinol stabilization",
        project_description="Research microencapsulation and oxidation stability for retinol serums.",
    )
    assert "retinol" in anchors
    assert "encapsulation" in anchors or "microencapsulation" in anchors
    assert "skincare" not in anchors


def test_social_query_terms_excludes_generic_project_words() -> None:
    """Social query construction should drop generic control words from fallback intent."""
    query = _query_for_social(
        {
            "search_filters": {"social": "#find #highly #relevant #social #retinol"},
            "keywords": ["find", "highly", "relevant", "retinol", "niacinamide", "stability"],
            "entities": [],
        },
        project_title="Find highly relevant social evidence",
        project_description="Track retinol niacinamide stability and compatibility",
    )
    lowered = query.lower()
    assert "retinol" in lowered
    assert "niacinamide" in lowered or "stability" in lowered
    assert "find" not in lowered
    assert "highly" not in lowered
    assert "relevant" not in lowered
