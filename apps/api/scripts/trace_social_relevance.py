"""Trace social relevance flow for Reddit/YouTube/X ingestion.

Runs sample project descriptions through the same ingest-layer functions used
in production, with mocked provider responses, so relevance filtering and
ranking behavior can be inspected end-to-end.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

# Provide a lightweight stub for ensembledata so ingest imports succeed in
# environments where the SDK is not installed.
ensemble_module = ModuleType("ensembledata")
ensemble_api_module = ModuleType("ensembledata.api")


class _DummyEDClient:  # pragma: no cover - trace harness stub
    def __init__(self, *args, **kwargs) -> None:
        _ = (args, kwargs)


ensemble_api_module.EDClient = _DummyEDClient  # type: ignore[attr-defined]
sys.modules.setdefault("ensembledata", ensemble_module)
sys.modules.setdefault("ensembledata.api", ensemble_api_module)


@dataclass
class SampleProject:
    title: str
    description: str
    intent: dict


SAMPLES = [
    SampleProject(
        title="Retinol Stability Research",
        description="Track user conversations, videos, and social posts on retinol stability and niacinamide compatibility for skincare formulations.",
        intent={
            "keywords": ["retinol", "niacinamide", "skincare formulation", "stability"],
            "must_match_terms": ["retinol", "niacinamide"],
            "query_specificity": "specific",
            "search_filters": {
                "social": "retinol niacinamide skincare stability",
                "instagram": "retinol skincare",
            },
        },
    ),
    SampleProject(
        title="Fragrance Longevity Trends",
        description="Understand long-lasting perfume trends and customer pain points around projection, sillage, and longevity.",
        intent={
            "keywords": ["fragrance", "perfume longevity", "projection", "sillage"],
            "must_match_terms": ["perfume", "longevity"],
            "query_specificity": "specific",
            "search_filters": {
                "social": "perfume longevity projection sillage",
                "instagram": "fragrance trends",
            },
        },
    ),
    SampleProject(
        title="Haircare Protein Build-up",
        description="Track social chatter on protein overload, bond repair, and wash routine cadence for curly hair.",
        intent={
            "keywords": ["protein overload", "bond repair", "curly hair", "hair routine"],
            "must_match_terms": ["protein", "bond repair"],
            "query_specificity": "specific",
            "search_filters": {
                "social": "protein overload bond repair curly hair",
                "instagram": "curly hair repair",
            },
        },
    ),
    SampleProject(
        title="Niche Brand Monitor",
        description="Monitor how users discuss @theordinary and niacinamide 10 around irritation and layering.",
        intent={
            "keywords": ["the ordinary", "niacinamide 10", "layering", "irritation"],
            "must_match_terms": ["niacinamide", "layering"],
            "query_specificity": "specific",
            "search_filters": {
                "social": "@theordinary niacinamide layering irritation",
                "instagram": "the ordinary niacinamide",
            },
        },
    ),
]


async def _mock_search_x_posts(query: str, max_results: int = 20) -> list[dict]:
    _ = max_results
    return [
        {
            "source_id": "x-relevant-1",
            "content": f"New thread on {query}: retinol with niacinamide routines",
            "author": "derm_chemist",
            "likes": 420,
            "retweets": 88,
            "replies": 34,
            "quotes": 12,
            "published_at": "2026-02-20T10:00:00+00:00",
            "url": "https://x.com/derm_chemist/status/1001",
        },
        {
            "source_id": "x-irrelevant-1",
            "content": "Football transfer rumours and match predictions",
            "author": "sports_daily",
            "likes": 2000,
            "retweets": 500,
            "replies": 120,
            "quotes": 20,
            "published_at": "2026-02-24T10:00:00+00:00",
            "url": "https://x.com/sports_daily/status/2001",
        },
    ]


async def _mock_search_reddit_posts(query: str, limit: int = 24, time_filter: str = "week") -> list[dict]:
    _ = (limit, time_filter)
    return [
        {
            "source_id": "reddit-relevant-1",
            "title": "Retinol + niacinamide routine results",
            "content": f"Detailed discussion on {query} and irritation management.",
            "author": "skin_formulator",
            "subreddit": "SkincareAddiction",
            "score": 540,
            "comments": 81,
            "published_at": "2026-02-21T08:00:00+00:00",
            "url": "https://www.reddit.com/r/SkincareAddiction/comments/abc123",
        },
        {
            "source_id": "reddit-irrelevant-1",
            "title": "Best PC gaming chair",
            "content": "Need recommendations for back support and arm rests.",
            "author": "desk_setup",
            "subreddit": "buildapc",
            "score": 900,
            "comments": 300,
            "published_at": "2026-02-24T08:00:00+00:00",
            "url": "https://www.reddit.com/r/buildapc/comments/xyz987",
        },
    ]


async def _mock_search_youtube_videos(query: str, max_results: int = 20) -> list[dict]:
    _ = max_results
    return [
        {
            "source_id": "yt-relevant-1",
            "title": "Retinol and Niacinamide Guide",
            "content": f"Practical explanation of {query} with evidence and routine design.",
            "author": "Skincare Lab",
            "thumbnail_url": "https://img.youtube.com/vi/relevant/default.jpg",
            "url": "https://www.youtube.com/watch?v=relevant123",
            "views": 120000,
            "likes": 7200,
            "comments": 530,
            "published_at": "2026-02-19T09:00:00+00:00",
        },
        {
            "source_id": "yt-irrelevant-1",
            "title": "Premier League Highlights",
            "content": "Top goals and saves from this week.",
            "author": "Football Channel",
            "thumbnail_url": "https://img.youtube.com/vi/irrelevant/default.jpg",
            "url": "https://www.youtube.com/watch?v=irrelevant456",
            "views": 450000,
            "likes": 12000,
            "comments": 900,
            "published_at": "2026-02-23T09:00:00+00:00",
        },
    ]


async def run_trace() -> None:
    ip = import_module("app.tasks.ingest_project")
    # Patch provider calls so traces are deterministic and do not depend on network/tokens.
    ip.search_x_posts = _mock_search_x_posts  # type: ignore[assignment]
    ip.search_reddit_posts = _mock_search_reddit_posts  # type: ignore[assignment]
    ip.search_youtube_videos = _mock_search_youtube_videos  # type: ignore[assignment]

    print("=== SOCIAL RELEVANCE TRACE ===")
    for idx, sample in enumerate(SAMPLES, start=1):
        print(f"\n--- SAMPLE {idx} ---")
        print(f"Initial project description: {sample.description}")
        query_terms = ip._social_query_terms(
            sample.intent,
            social_filter=str(sample.intent.get("search_filters", {}).get("social", "")),
            project_title=sample.title,
            project_description=sample.description,
        )
        print(f"Derived social query terms: {query_terms}")
        derived_query = ip._query_for_social(
            sample.intent,
            social_filter="",
            project_title=sample.title,
            project_description=sample.description,
        )
        print(f"Derived social query: {derived_query}")

        raw_x = await _mock_search_x_posts(derived_query)
        raw_reddit = await _mock_search_reddit_posts(derived_query)
        raw_youtube = await _mock_search_youtube_videos(derived_query)

        x_docs = await ip._ingest_x(
            project_id=f"proj-{idx}",
            user_id="user-trace",
            intent=sample.intent,
            social_filter="",
            project_title=sample.title,
            project_description=sample.description,
        )
        reddit_docs = await ip._ingest_reddit(
            project_id=f"proj-{idx}",
            user_id="user-trace",
            intent=sample.intent,
            social_filter="",
            project_title=sample.title,
            project_description=sample.description,
        )
        youtube_docs = await ip._ingest_youtube(
            project_id=f"proj-{idx}",
            user_id="user-trace",
            intent=sample.intent,
            social_filter="",
            project_title=sample.title,
            project_description=sample.description,
        )

        print(f"X kept docs: {len(x_docs)}")
        print(f"X fetched raw items: {len(raw_x)} | dropped: {len(raw_x) - len(x_docs)}")
        for doc in x_docs:
            print(f"  - {doc.source_id} | title={doc.title} | url={doc.metadata.get('url')}")
        print(f"Reddit kept docs: {len(reddit_docs)}")
        print(
            f"Reddit fetched raw items: {len(raw_reddit)} | dropped: {len(raw_reddit) - len(reddit_docs)}"
        )
        for doc in reddit_docs:
            print(f"  - {doc.source_id} | title={doc.title} | url={doc.metadata.get('url')}")
        print(f"YouTube kept docs: {len(youtube_docs)}")
        print(
            f"YouTube fetched raw items: {len(raw_youtube)} | dropped: {len(raw_youtube) - len(youtube_docs)}"
        )
        for doc in youtube_docs:
            print(f"  - {doc.source_id} | title={doc.title} | url={doc.metadata.get('url')}")

        print("Relevance summary:")
        print(
            "  - X/Reddit/YouTube items must pass lexical intent matching in ingest "
            f"(min matches: {ip.settings.INGEST_SOCIAL_MIN_RELEVANCE_MATCHES})."
        )
        print("  - Remaining items are reranked by relevance + engagement + recency.")

    print("\n--- EDGE CASE CHECKS ---")
    print("1) Provider returns malformed/empty items")

    async def _x_malformed(query: str, max_results: int = 20) -> list[dict]:
        _ = (query, max_results)
        return [
            {"source_id": "", "content": "missing id"},
            {"source_id": "bad-x-1", "content": ""},
            {"source_id": "bad-x-2", "content": "football highlights"},
        ]

    async def _reddit_malformed(query: str, limit: int = 24, time_filter: str = "week") -> list[dict]:
        _ = (query, limit, time_filter)
        return [
            {"source_id": "", "title": "No id"},
            {"source_id": "bad-r-1", "title": "", "content": ""},
            {"source_id": "bad-r-2", "title": "Gaming chair", "content": "ergonomic setup"},
        ]

    async def _youtube_malformed(query: str, max_results: int = 20) -> list[dict]:
        _ = (query, max_results)
        return [
            {"source_id": "", "title": "No id"},
            {"source_id": "bad-y-1", "title": "", "content": ""},
            {"source_id": "bad-y-2", "title": "Football clips", "content": "goal highlights"},
        ]

    ip.search_x_posts = _x_malformed  # type: ignore[assignment]
    ip.search_reddit_posts = _reddit_malformed  # type: ignore[assignment]
    ip.search_youtube_videos = _youtube_malformed  # type: ignore[assignment]

    edge_intent = {
        "keywords": ["retinol", "niacinamide"],
        "must_match_terms": ["retinol"],
        "query_specificity": "specific",
        "search_filters": {"social": "retinol niacinamide"},
    }
    edge_x = await ip._ingest_x(
        project_id="edge-proj",
        user_id="edge-user",
        intent=edge_intent,
        social_filter="",
        project_title="Edge Project",
        project_description="Edge description",
    )
    edge_reddit = await ip._ingest_reddit(
        project_id="edge-proj",
        user_id="edge-user",
        intent=edge_intent,
        social_filter="",
        project_title="Edge Project",
        project_description="Edge description",
    )
    edge_youtube = await ip._ingest_youtube(
        project_id="edge-proj",
        user_id="edge-user",
        intent=edge_intent,
        social_filter="",
        project_title="Edge Project",
        project_description="Edge description",
    )
    print(
        "Malformed-item results (should be 0 kept without crashing): "
        f"X={len(edge_x)}, Reddit={len(edge_reddit)}, YouTube={len(edge_youtube)}"
    )


if __name__ == "__main__":
    asyncio.run(run_trace())
