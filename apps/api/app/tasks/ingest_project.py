"""ingest_project ARQ task — initial KB population from enabled sources after project creation.

Called immediately after project creation via ARQ enqueue.
Also re-used by refresh_project for subsequent refreshes.

Supports three independently toggleable ingestion sources per project:
  - Instagram  (instagram_enabled)
  - TikTok     (tiktok_enabled)
  - OpenAlex   (openalex_enabled)

Partial success is intentional: one source failing never aborts the others.
"""

import asyncio
import json
import math
import re
from datetime import UTC, datetime

import structlog

from app.core.cache_version import bump_project_cache_version
from app.core.config import settings
from app.core.db import get_db_pool
from app.core.redis import get_redis
from app.kb.ingester import RawDocument, ingest_documents
from app.repositories import project as project_repo
from app.tools.papers_openalex import fetch_papers
from app.tools.social_instagram import (
    instagram_hashtag_posts,
    instagram_post_info,
    instagram_search_multi,
    instagram_user_basic_stats,
    instagram_user_reels,
)
from app.tools.social_tiktok import fetch_tiktok_posts

logger = structlog.get_logger(__name__)


async def _invalidate_project_caches(project_id: str) -> None:
    """
    Invalidate all caches for a project after KB update.

    Bumps project cache version so semantic + KB hot cache keys rotate immediately,
    then clears project-scoped search cache keys via SCAN.

    Cache invalidation failure is logged but does not crash the ingestion
    task — data consistency is maintained even if caches remain stale.
    """
    try:
        redis = await get_redis()
        new_version = await bump_project_cache_version(redis, project_id)
        patterns = [
            f"search:cache:{project_id}:*",
        ]

        deleted_count = 0
        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis.delete(*keys)
                    deleted_count += len(keys)
                if cursor == 0:
                    break

        logger.info(
            "cache_invalidation.complete",
            project_id=project_id,
            cache_version=new_version,
            keys_deleted=deleted_count,
        )
    except Exception:
        logger.exception(
            "cache_invalidation.failed",
            project_id=project_id,
        )


ACCOUNT_RELEVANCE_WEIGHT = 6.0
ACCOUNT_BRAND_WEIGHT = 5.0
ACCOUNT_CREDIBILITY_WEIGHT = 2.0
ACCOUNT_REACH_WEIGHT = 3.0
ACCOUNT_SCORE_WEIGHT = 1.0
REEL_RELEVANCE_WEIGHT = 6.0
REEL_BRAND_WEIGHT = 5.0
REEL_ENGAGEMENT_WEIGHT = 8.0
REEL_RECENCY_WEIGHT = 2.0
REEL_RECENCY_DAYS = 30.0

_BEAUTY_SCOPE_TERMS: set[str] = {
    "beauty",
    "cosmetic",
    "cosmetics",
    "skincare",
    "skin",
    "makeup",
    "fragrance",
    "perfume",
    "haircare",
    "moisturizer",
    "sunscreen",
    "balm",
    "lipbalm",
    "lip",
    "lipstick",
    "lipgloss",
    "concealer",
    "foundation",
    "serum",
    "cleanser",
    "toner",
    "exfoliant",
    "acne",
    "pigmentation",
    "bodycare",
    "retinol",
    "niacinamide",
    "hyaluronic",
    "peptides",
    "ceramide",
    "aha",
    "bha",
}

_GENERIC_RELEVANCE_TERMS: set[str] = {
    "trend",
    "trends",
    "research",
    "analysis",
    "market",
    "industry",
    "product",
    "products",
    "news",
    "viral",
}


def _as_int(value: object) -> int:
    """Convert mixed numeric payload values (int/float/str) to int safely."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return 0
        try:
            return int(float(cleaned))
        except ValueError:
            return 0
    return 0


def _tokenize(text: str) -> set[str]:
    """Lowercase token set for keyword relevance checks."""
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 3}


def _keyword_tokens(text: str) -> set[str]:
    """Tokenize keywords and add a compact alphanumeric form for brand-like terms."""
    tokens = _tokenize(text)
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(compact) >= 3:
        tokens.add(compact)
    return tokens


def _contains_beauty_signal(text: str) -> bool:
    """Return True if text includes at least one beauty-domain token."""
    return len(_tokenize(text) & _BEAUTY_SCOPE_TERMS) > 0


def _filter_relevance_terms(terms: set[str]) -> set[str]:
    """Remove low-signal generic tokens while preserving beauty-domain terms."""
    filtered = {t for t in terms if t and t not in _GENERIC_RELEVANCE_TERMS}
    return filtered


def _build_relevance_terms(intent: dict, social_filter: str) -> set[str]:
    """
    Build keyword terms used to rank Instagram accounts and reels.

    Terms come from extracted keywords, instagram search filter, and social hashtags.
    """
    terms: set[str] = set()
    keywords: list[str] = intent.get("keywords") or []
    for kw in keywords:
        terms.update(_keyword_tokens(str(kw)))

    search_filters = intent.get("search_filters") or {}
    terms.update(_keyword_tokens(str(search_filters.get("instagram") or "")))
    terms.update(_keyword_tokens(str(social_filter or "")))
    terms = _filter_relevance_terms(terms)
    if not terms:
        terms = {"beauty", "cosmetics", "skincare", "makeup", "fragrance"}
    return terms


def _build_brand_terms(intent: dict) -> set[str]:
    """
    Extract non-generic brand/product tokens from intent keywords.

    These are used as a boost signal so projects like P&G/Olay/SK-II do not
    get dominated by generic 'beauty' accounts.
    """
    keywords: list[str] = intent.get("keywords") or []
    terms: set[str] = set()
    for kw in keywords:
        for tok in _keyword_tokens(str(kw)):
            if tok in _GENERIC_RELEVANCE_TERMS:
                continue
            if tok in _BEAUTY_SCOPE_TERMS:
                continue
            terms.add(tok)
    return terms


def _match_count(text: str, terms: set[str]) -> int:
    """Count distinct relevance-term matches in text."""
    if not text or not terms:
        return 0
    tokens = _tokenize(text)
    return len(tokens & terms)


def _clean_instagram_keyword_query(raw: str) -> str:
    """
    Convert a raw keyword into a safe short Instagram search query.

    Keeps only letters/numbers/spaces, collapses whitespace, and limits to
    max 3 words to match the Ensemble Instagram user-search behavior.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    words = cleaned.split()[:3]
    if not words:
        return ""
    return " ".join(words)


def _extract_hashtags(text: str) -> list[str]:
    """Extract cleaned hashtag terms from a social filter string."""
    if not text:
        return []
    tags = re.findall(r"#([a-zA-Z0-9_]+)", text)
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = _clean_instagram_keyword_query(tag.replace("_", " "))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


async def ingest_project(ctx: dict, project_id: str) -> None:
    """
    Populate a project's KB from all enabled sources based on its structured_intent.

    Flow:
    1. Fetch project row including per-source toggle columns
    2. Validate social_filter from structured_intent.search_filters.social
    3. Fan out to each enabled source (Instagram, TikTok, OpenAlex) independently
    4. Build RawDocument list from all collected content
    5. ingest_documents() → chunk, embed, upsert to knowledge_chunks
    6. Update kb_chunk_count and last_refreshed_at

    oldest_timestamp=None means fetch all available content (initial run).
    For refresh runs, pass oldest_timestamp via refresh_project instead.
    """
    await _run_ingestion(ctx, project_id, oldest_timestamp=None)


async def _run_ingestion(
    ctx: dict,
    project_id: str,
    oldest_timestamp: int | None,
) -> None:
    """
    Core ingestion logic shared by ingest_project and refresh_project.

    Fans out to all enabled sources independently.  A source returning an empty
    list is treated as a warning, not a fatal error — other sources continue.
    Separated so refresh_project can pass oldest_timestamp without duplicating
    the full ingestion flow.
    """
    pool = await get_db_pool()

    # Fetch project without ownership check — workers run as superuser
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, user_id::text, structured_intent, last_refreshed_at,
                   tiktok_enabled, instagram_enabled, openalex_enabled
            FROM projects
            WHERE id = $1::uuid
            """,
            project_id,
        )

    if not row:
        logger.warning("ingest_project.project_not_found", project_id=project_id)
        return

    project = dict(row)
    raw_intent = project.get("structured_intent") or {}
    # asyncpg returns jsonb columns as str — decode if needed
    if isinstance(raw_intent, str):
        try:
            raw_intent = json.loads(raw_intent)
        except (json.JSONDecodeError, ValueError):
            raw_intent = {}
    intent = dict(raw_intent)
    user_id = project["user_id"]
    social_filter = (intent.get("search_filters") or {}).get("social", "")

    instagram_enabled: bool = bool(project.get("instagram_enabled"))
    tiktok_enabled: bool = bool(project.get("tiktok_enabled"))
    openalex_enabled: bool = bool(project.get("openalex_enabled"))

    if not social_filter:
        logger.warning("ingest_project.no_social_filter", project_id=project_id)
        # Disable social sources — they need a filter — but do NOT return.
        # OpenAlex can still run independently using search_filters.papers.
        instagram_enabled = False
        tiktok_enabled = False

    logger.info(
        "ingest_project.start",
        project_id=project_id,
        social_filter=social_filter[:60],
        instagram_enabled=instagram_enabled,
        tiktok_enabled=tiktok_enabled,
        openalex_enabled=openalex_enabled,
    )

    documents: list[RawDocument] = []

    # ------------------------------------------------------------------ #
    # Source 1: Instagram                                                  #
    # ------------------------------------------------------------------ #
    if instagram_enabled:
        try:
            instagram_docs = await _ingest_instagram(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
                oldest_timestamp=oldest_timestamp,
            )
            documents.extend(instagram_docs)
        except Exception:
            logger.warning(
                "ingest_project.instagram_fan.failed",
                project_id=project_id,
            )

    # ------------------------------------------------------------------ #
    # Source 2: TikTok                                                     #
    # ------------------------------------------------------------------ #
    if tiktok_enabled:
        try:
            tiktok_docs = await _ingest_tiktok(
                project_id=project_id,
                user_id=user_id,
                social_filter=social_filter,
            )
            documents.extend(tiktok_docs)
        except Exception:
            logger.warning(
                "ingest_project.tiktok_fan.failed",
                project_id=project_id,
            )

    # ------------------------------------------------------------------ #
    # Source 3: OpenAlex                                                   #
    # ------------------------------------------------------------------ #
    if openalex_enabled:
        try:
            paper_docs = await _ingest_openalex(
                project_id=project_id,
                user_id=user_id,
                intent=intent,
                social_filter=social_filter,
            )
            documents.extend(paper_docs)
        except Exception:
            logger.warning(
                "ingest_project.openalex_fan.failed",
                project_id=project_id,
            )

    # ------------------------------------------------------------------ #
    # Nothing collected across all enabled sources                         #
    # ------------------------------------------------------------------ #
    if not documents:
        logger.warning("ingest_project.no_documents", project_id=project_id)
        # Still update refreshed_at so we don't re-run immediately
        await project_repo.update_project_kb_stats(pool, project_id, 0, datetime.now(UTC))
        return

    inserted = await ingest_documents(documents)

    # Get updated total chunk count
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE project_id = $1::uuid",
            project_id,
        )

    await project_repo.update_project_kb_stats(pool, project_id, int(total or 0), datetime.now(UTC))

    instagram_docs_count = sum(1 for d in documents if d.source == "social_instagram")
    tiktok_docs_count = sum(1 for d in documents if d.source == "social_tiktok")
    paper_docs_count = sum(1 for d in documents if d.source == "paper")

    logger.info(
        "ingest_project.done",
        project_id=project_id,
        instagram_docs=instagram_docs_count,
        tiktok_docs=tiktok_docs_count,
        paper_docs=paper_docs_count,
        total_docs=len(documents),
        chunks_inserted=inserted,
        total_chunks=total,
    )


async def _ingest_instagram(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
    oldest_timestamp: int | None,
) -> list[RawDocument]:
    """
    Discover Instagram accounts for the project and collect their posts.

    Search + ranking strategy:
    1. Build multiple short user-search queries from:
       - search_filters.instagram
       - intent keywords
       - hashtag terms from search_filters.social, with keyword fallback
    2. Run instagram_search_multi across all queries and deduplicate accounts.
    3. Rank accounts by relevance + credibility + reach.
    4. Fetch reels for the top ranked accounts and score reels globally by:
       caption relevance + engagement + recency + account score.
       Additionally, fetch hashtag posts via EnsembleData's hashtag endpoint
       (direct HTTP call) to improve keyword-level reel coverage.
    5. Select final reels with per-account caps for diversity.

    Returns a list of RawDocument with source="social_instagram".
    Never raises — all exceptions are caught and logged.
    """
    search_filters = intent.get("search_filters") or {}
    instagram_query = (search_filters.get("instagram") or "").strip()

    keywords: list[str] = intent.get("keywords") or []
    short_kw: list[str] = []
    seen_short_kw: set[str] = set()
    for keyword in keywords:
        safe_query = _clean_instagram_keyword_query(str(keyword))
        if not safe_query:
            continue
        if len(safe_query.split()) > 3:
            continue
        key = safe_query.lower()
        if key in seen_short_kw:
            continue
        seen_short_kw.add(key)
        short_kw.append(safe_query)
        if len(short_kw) >= 7:
            break

    hashtags = _extract_hashtags(social_filter)
    search_queries: list[str] = []
    seen_search_queries: set[str] = set()
    if instagram_query:
        clean_primary = _clean_instagram_keyword_query(instagram_query)
        if clean_primary:
            search_queries.append(clean_primary)
            seen_search_queries.add(clean_primary.lower())
    for query in [*short_kw, *hashtags]:
        key = query.lower()
        if key in seen_search_queries:
            continue
        seen_search_queries.add(key)
        search_queries.append(query)

    search_results: list[dict] = await instagram_search_multi(search_queries)
    logger.info(
        "ingest_project.instagram.search_queries",
        project_id=project_id,
        query_count=len(search_queries),
        query_preview=search_queries[:6],
        result_count=len(search_results),
    )

    if not search_results:
        logger.warning(
            "ingest_project.instagram.no_accounts",
            project_id=project_id,
            instagram_query=instagram_query,
        )
        return []

    relevance_terms = _build_relevance_terms(intent, social_filter)
    brand_terms = _build_brand_terms(intent)

    async def _ranked_accounts(results: list[dict]) -> list[dict]:
        candidates = results[: settings.INGEST_INSTAGRAM_ACCOUNT_CANDIDATES]
        prepared: list[tuple[dict, int, str, int, int, int]] = []
        for account in candidates:
            raw_user_id = account.get("pk") or account.get("id")
            if not raw_user_id:
                continue
            try:
                uid = int(str(raw_user_id))
            except (TypeError, ValueError):
                continue

            username = str(account.get("username") or account.get("handle") or "")
            full_name = str(account.get("full_name") or "")
            bio = str(
                account.get("biography")
                or account.get("bio")
                or account.get("biography_with_entities")
                or ""
            )
            text = f"{username} {full_name} {bio}".strip()
            relevance = _match_count(text, relevance_terms)
            brand_matches = _match_count(text, brand_terms)
            if (
                relevance < settings.INGEST_INSTAGRAM_MIN_ACCOUNT_RELEVANCE
                and brand_matches == 0
                and not _contains_beauty_signal(text)
            ):
                continue
            is_verified = 1 if bool(account.get("is_verified")) else 0
            prepared.append((account, uid, username, relevance, brand_matches, is_verified))

        stats_tasks = [instagram_user_basic_stats(uid) for _, uid, _, _, _, _ in prepared]
        stats_results = await asyncio.gather(*stats_tasks, return_exceptions=True)

        ranked: list[dict] = []
        for (account, uid, username, relevance, brand_matches, is_verified), stats in zip(
            prepared,
            stats_results,
            strict=False,
        ):
            if isinstance(stats, Exception):
                logger.warning(
                    "ingest_project.instagram.stats_fetch_failed",
                    project_id=project_id,
                    uid=uid,
                    username=username,
                    error_type=type(stats).__name__,
                )
                stats = {}
            stats_dict = stats if isinstance(stats, dict) else {}

            followers = _as_int(stats_dict.get("followers"))

            score = (
                (relevance * ACCOUNT_RELEVANCE_WEIGHT)
                + (brand_matches * ACCOUNT_BRAND_WEIGHT)
                + (is_verified * ACCOUNT_CREDIBILITY_WEIGHT)
                + (math.log10(1 + followers) * ACCOUNT_REACH_WEIGHT)
            )
            ranked.append(
                {
                    "account": account,
                    "uid": uid,
                    "username": username or str(uid),
                    "followers": followers,
                    "score": score,
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[: settings.INGEST_MAX_ACCOUNTS]

    selected_accounts = await _ranked_accounts(search_results)
    if not selected_accounts:
        logger.warning("ingest_project.instagram.no_ranked_accounts", project_id=project_id)
        return []

    accounts_to_fetch = min(
        max(1, settings.INGEST_INSTAGRAM_ACCOUNTS_TO_FETCH),
        max(1, len(selected_accounts)),
    )
    candidates = selected_accounts[:accounts_to_fetch]

    reels_results = await asyncio.gather(
        *[
            instagram_user_reels(s["uid"], depth=1, oldest_timestamp=oldest_timestamp)
            for s in candidates
        ],
        return_exceptions=True,
    )

    scored_reels_with_account: list[tuple[float, dict, str, str, int]] = []
    scored_reels_fallback: list[tuple[float, dict, str, str, int]] = []
    now_ts = datetime.now(UTC).timestamp()
    hashtag_posts_total = 0

    for selected, reels in zip(candidates, reels_results, strict=False):
        uid = selected["uid"]
        account_key = f"uid:{uid}"
        username = selected["username"]
        followers = selected["followers"]
        account_score = float(selected.get("score") or 0.0)

        if isinstance(reels, Exception):
            logger.warning(
                "ingest_project.instagram.reels_fetch_failed",
                project_id=project_id,
                uid=uid,
                username=username,
                error_type=type(reels).__name__,
            )
            continue
        reels_list = reels if isinstance(reels, list) else []

        logger.info(
            "ingest_project.instagram.reels_fetched",
            project_id=project_id,
            username=username,
            reel_count=len(reels_list),
            followers=followers,
        )

        # Enrich a small number of reels with post-details when view count is missing.
        # Ensemble user_reels often omits reliable view counts on older media.
        enrichable = [
            r
            for r in reels_list
            if _as_int(r.get("view_count")) <= 0 and (r.get("shortcode") or "")
        ]
        enrichable.sort(key=lambda r: _as_int(r.get("like_count")), reverse=True)
        for reel in enrichable[: settings.INGEST_INSTAGRAM_DETAIL_ENRICH_PER_ACCOUNT]:
            shortcode = str(reel.get("shortcode") or "")
            try:
                details = await instagram_post_info(shortcode)
            except Exception as exc:
                logger.warning(
                    "ingest_project.instagram.reel_detail_fetch_failed",
                    project_id=project_id,
                    uid=uid,
                    username=username,
                    shortcode=shortcode,
                    error_type=type(exc).__name__,
                )
                continue
            if details:
                reel["like_count"] = _as_int(details.get("like_count")) or _as_int(
                    reel.get("like_count")
                )
                reel["view_count"] = _as_int(details.get("view_count")) or _as_int(
                    reel.get("view_count")
                )
                if not reel.get("video_url"):
                    reel["video_url"] = details.get("video_url") or ""
                if not reel.get("cover_url"):
                    reel["cover_url"] = details.get("cover_url") or ""
        for reel in reels_list:
            caption = (reel.get("caption") or "").strip()
            if not caption:
                continue
            if not _contains_beauty_signal(caption):
                continue
            likes = _as_int(reel.get("like_count"))
            views = _as_int(reel.get("view_count"))
            relevance = _match_count(caption, relevance_terms)
            brand_match = _match_count(caption, brand_terms)
            engagement = math.log1p(max(0, likes + (views / 10)))

            ts_value = reel.get("timestamp")
            age_days = REEL_RECENCY_DAYS
            if isinstance(ts_value, int | float) and ts_value > 0:
                age_days = max(0.0, (now_ts - float(ts_value)) / 86400.0)
            recency = math.exp(-age_days / REEL_RECENCY_DAYS)

            score = (
                (relevance * REEL_RELEVANCE_WEIGHT)
                + (brand_match * REEL_BRAND_WEIGHT)
                + (engagement * REEL_ENGAGEMENT_WEIGHT)
                + (recency * REEL_RECENCY_WEIGHT)
                + (math.log1p(max(0.0, account_score)) * ACCOUNT_SCORE_WEIGHT)
            )
            packed = (score, reel, account_key, username, followers)
            scored_reels_fallback.append(packed)
            if relevance >= settings.INGEST_INSTAGRAM_MIN_REEL_RELEVANCE or brand_match > 0:
                scored_reels_with_account.append(packed)

    max_hashtag_queries = max(1, settings.INGEST_INSTAGRAM_HASHTAG_QUERIES)
    hashtag_queries: list[str] = []
    seen_hashtag_queries: set[str] = set()
    for candidate in [*hashtags, *short_kw]:
        cleaned = _clean_instagram_keyword_query(candidate)
        if not cleaned:
            continue
        normalized = cleaned.lower()
        if normalized in seen_hashtag_queries:
            continue
        if (
            _match_count(cleaned, brand_terms) == 0
            and _match_count(cleaned, relevance_terms) == 0
            and not _contains_beauty_signal(cleaned)
        ):
            continue
        seen_hashtag_queries.add(normalized)
        hashtag_queries.append(cleaned)
        if len(hashtag_queries) >= max_hashtag_queries:
            break
    for hashtag in hashtag_queries:
        cursor: str | None = None
        for _ in range(max(1, settings.INGEST_INSTAGRAM_HASHTAG_PAGES)):
            hashtag_posts, next_cursor = await instagram_hashtag_posts(
                hashtag=hashtag.replace(" ", ""),
                cursor=cursor,
                get_author_info=True,
            )
            hashtag_posts_total += len(hashtag_posts)
            for post in hashtag_posts:
                caption = (post.get("caption") or "").strip()
                if not caption or not _contains_beauty_signal(caption):
                    continue
                if not str(post.get("video_url") or "").strip():
                    continue

                likes = _as_int(post.get("like_count"))
                views = _as_int(post.get("view_count"))
                relevance = _match_count(caption, relevance_terms)
                brand_match = _match_count(caption, brand_terms)
                engagement = math.log1p(max(0, likes + (views / 10)))
                ts_value = post.get("timestamp")
                age_days = REEL_RECENCY_DAYS
                if isinstance(ts_value, int | float) and ts_value > 0:
                    age_days = max(0.0, (now_ts - float(ts_value)) / 86400.0)
                recency = math.exp(-age_days / REEL_RECENCY_DAYS)

                score = (
                    (relevance * REEL_RELEVANCE_WEIGHT)
                    + (brand_match * REEL_BRAND_WEIGHT)
                    + (engagement * REEL_ENGAGEMENT_WEIGHT)
                    + (recency * REEL_RECENCY_WEIGHT)
                )
                username = str(post.get("username") or "").strip() or "instagram"
                account_key = f"user:{username.lower()}"
                packed = (score, post, account_key, username, 0)
                scored_reels_fallback.append(packed)
                if relevance >= settings.INGEST_INSTAGRAM_MIN_REEL_RELEVANCE or brand_match > 0:
                    scored_reels_with_account.append(packed)

            cursor = next_cursor
            if not cursor:
                break

    ranked_pool = scored_reels_with_account or scored_reels_fallback
    ranked_pool.sort(key=lambda x: x[0], reverse=True)

    docs: list[RawDocument] = []
    per_account_counts: dict[str, int] = {}
    seen_source_ids: set[str] = set()
    max_per_account = max(1, settings.INGEST_INSTAGRAM_MAX_REELS_PER_ACCOUNT)
    global_limit = max(1, settings.INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT)
    for _, reel, account_key, username, followers in ranked_pool:
        if per_account_counts.get(account_key, 0) >= max_per_account:
            continue

        code = reel.get("shortcode") or ""
        if not code or code in seen_source_ids:
            continue
        caption = (reel.get("caption") or "").strip()
        if not caption:
            continue
        if not str(reel.get("video_url") or "").strip():
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_instagram",
                source_id=code,
                title=f"@{username}",
                content=caption,
                metadata={
                    "platform": "instagram",
                    "username": username,
                    "followers": followers,
                    "like_count": reel.get("like_count") or 0,
                    "view_count": reel.get("view_count") or 0,
                    "timestamp": reel.get("timestamp"),
                    "cover_url": reel.get("cover_url", ""),
                    "video_url": reel.get("video_url", ""),
                    "url": f"https://www.instagram.com/reel/{code}/" if code else None,
                },
            )
        )
        seen_source_ids.add(code)
        per_account_counts[account_key] = per_account_counts.get(account_key, 0) + 1
        if len(docs) >= global_limit:
            break

    logger.info(
        "ingest_project.instagram.reels_selected",
        project_id=project_id,
        selected_docs=len(docs),
        candidate_reels=len(ranked_pool),
        hashtag_posts=hashtag_posts_total,
        used_relevance_pool=bool(scored_reels_with_account),
    )
    return docs


async def _ingest_tiktok(
    *,
    project_id: str,
    user_id: str,
    social_filter: str,
) -> list[RawDocument]:
    """
    Fetch recent TikTok posts for the handle derived from social_filter.

    social_filter is reused as the TikTok handle (same structured_intent field).
    Videos with an empty description are skipped — they carry no KB value.

    Returns a list of RawDocument with source="social_tiktok".
    Returns [] (and logs a warning) if the fetch yields no posts.
    Never raises — all exceptions are caught and logged.
    """
    handle = social_filter
    posts = await fetch_tiktok_posts(handle)
    logger.info(
        "ingest_project.tiktok.posts_fetched",
        project_id=project_id,
        handle=handle,
        post_count=len(posts),
    )

    docs: list[RawDocument] = []
    for video in posts:
        description = (video.get("description") or "").strip()
        if not description:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="social_tiktok",
                source_id=video.get("video_id", ""),
                title=f"@{video.get('author_username', '')}",
                content=description,
                metadata={
                    "platform": "tiktok",
                    "author_username": video.get("author_username", ""),
                    "likes": video.get("likes", 0),
                    "views": video.get("views", 0),
                    "cover_url": video.get("cover_url", ""),
                    "video_url": video.get("video_url", ""),
                },
            )
        )

    return docs


async def _ingest_openalex(
    *,
    project_id: str,
    user_id: str,
    intent: dict,
    social_filter: str,
) -> list[RawDocument]:
    """
    Fetch research papers from OpenAlex using the best available query string.

    Query priority:
    1. structured_intent.search_filters.papers — natural language academic query
    2. Fallback: social_filter with hashtag symbols stripped

    The old code read intent.get("description") / intent.get("query") — neither
    field exists in structured_intent — so it always fell through to the raw
    social_filter hashtag string (e.g. "#PG #beautytrends"), which OpenAlex
    cannot match. This fix reads the correct field.

    Papers with no abstract are skipped — an empty abstract produces useless KB chunks.

    Returns a list of RawDocument with source="paper".
    Returns [] (and logs a warning) if the fetch yields no papers.
    Never raises — all exceptions are caught and logged.
    """
    search_filters = intent.get("search_filters") or {}
    papers_query = (search_filters.get("papers") or "").strip()
    # Strip # symbols from any fallback — OpenAlex cannot match hashtag strings
    fallback = social_filter.replace("#", " ").strip()
    query = papers_query or fallback
    papers = await fetch_papers(query)
    logger.info(
        "ingest_project.openalex.papers_fetched",
        project_id=project_id,
        query_preview=query[:80],
        paper_count=len(papers),
    )

    docs: list[RawDocument] = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        docs.append(
            RawDocument(
                project_id=project_id,
                user_id=user_id,
                source="paper",
                source_id=paper.get("paper_id", ""),
                title=paper.get("title", ""),
                content=abstract,
                metadata={
                    "doi": paper.get("doi", ""),
                    "publication_year": paper.get("publication_year", 0),
                },
            )
        )

    return docs
