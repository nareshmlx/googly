"""ProjectService — orchestrates project CRUD, intent extraction, and ARQ enqueueing.

Rule: This file contains business logic only. No direct SQL. No HTTP.
DB access goes through app.repositories.project. ARQ enqueuing goes through
the ARQ pool. Intent extraction goes through app.kb.intent_extractor.
"""

import structlog
from arq import ArqRedis

from app.core.constants import RedisKeys
from app.core.db import get_db_pool
from app.core.redis import get_redis
from app.kb.intent_extractor import extract_intent
from app.repositories import project as project_repo

logger = structlog.get_logger(__name__)


async def _invalidate_projects_summary_cache(user_id: str) -> None:
    """Invalidate cached project summaries for one user."""
    try:
        redis = await get_redis()
        await redis.delete(RedisKeys.PROJECTS_SUMMARY.format(user_id=user_id))
    except Exception:
        logger.warning("project_service.projects_summary_cache_invalidate_failed", user_id=user_id)


async def create_project(
    user_id: str,
    title: str,
    description: str,
    refresh_strategy: str,
    arq_pool: ArqRedis,
    tiktok_enabled: bool = True,
    instagram_enabled: bool = True,
    openalex_enabled: bool = True,
) -> dict:
    """
    Create a project, extract structured intent, and enqueue initial KB ingestion.

    Intent extraction runs synchronously here (not in the worker) so the response
    to the client already contains meaningful structured_intent — the UI can show
    the extracted keywords immediately without polling.

    Enqueueing ingest_project is fire-and-forget: the worker runs after this
    returns. Project creation does not wait for KB population.
    """
    pool = await get_db_pool()

    logger.info("project_service.create.start", user_id=user_id, title=title)

    # 1. Insert with empty intent first — get a real project_id
    project = await project_repo.insert_project(
        pool,
        user_id,
        title,
        description,
        refresh_strategy,
        tiktok_enabled=tiktok_enabled,
        instagram_enabled=instagram_enabled,
        openalex_enabled=openalex_enabled,
    )
    project_id = project["id"]
    logger.info("project_service.create.inserted", project_id=project_id)

    # 2. Extract intent synchronously — safe to fail (returns default on error)
    intent = await extract_intent(description)
    logger.info(
        "project_service.create.intent_extracted",
        project_id=project_id,
        domain=intent.get("domain"),
    )

    # 3. Store intent
    await project_repo.update_project_intent(pool, project_id, intent)
    project["structured_intent"] = intent
    await _invalidate_projects_summary_cache(user_id)

    # 4. Enqueue initial KB population (non-blocking)
    try:
        await arq_pool.enqueue_job("ingest_project", project_id)
        logger.info("project_service.create.ingestion_enqueued", project_id=project_id)
    except Exception:
        # ARQ failure must not fail project creation — KB will be empty until user retries
        logger.warning("project_service.create.enqueue_failed", project_id=project_id)

    return project


async def list_projects(user_id: str) -> list[dict]:
    """Return all projects for a user, newest first."""
    pool = await get_db_pool()
    return await project_repo.list_projects(pool, user_id)


async def get_project(project_id: str, user_id: str) -> dict | None:
    """
    Fetch a single project scoped to its owner.

    Returns None if the project does not exist or the user does not own it.
    Caller is responsible for returning 404.
    """
    pool = await get_db_pool()
    return await project_repo.fetch_project(pool, project_id, user_id)


async def get_discover_feed(project_id: str) -> list[dict]:
    """Fetch social KB chunks for the discover feed (TikTok + Instagram only).

    Uses a platform-balanced strategy to prevent high-engagement TikTok items
    from crowding out all Instagram content, while also preventing single
    Instagram creators from dominating the feed:
      - Guaranteed slot: top 15 Instagram items by engagement×recency score
        with max 2 items per creator
      - Remaining 60 slots: best-scored items from either platform, excluding
        the 15 already selected Instagram items, and capped to max 3 items
        per Instagram creator
      - Total: up to 75 items

    Engagement × recency score:
        score = ln(1 + likes + views/10) × exp(-days_old / 30)

    Recency uses the content timestamp from metadata when available, falling
    back to DB created_at.

    Both TikTok field names (likes, views) and Instagram field names (like_count,
    view_count) are handled via COALESCE so the same formula covers both platforms.

    Returns an empty list when no social chunks exist — callers must not treat
    this as an error condition.
    """
    pool = await get_db_pool()
    rows = await project_repo.fetch_discover_feed(pool, project_id)
    logger.info(
        "project_service.discover_feed.fetched",
        project_id=project_id,
        total=len(rows),
        instagram=sum(1 for r in rows if r["source"] == "social_instagram"),
        tiktok=sum(1 for r in rows if r["source"] == "social_tiktok"),
    )
    return rows


async def delete_project(project_id: str, user_id: str) -> bool:
    """
    Delete a project and all its knowledge chunks (cascade).

    Returns True if deleted, False if not found / not owned.
    """
    pool = await get_db_pool()
    deleted = await project_repo.delete_project(pool, project_id, user_id)
    if deleted:
        logger.info("project_service.deleted", project_id=project_id, user_id=user_id)
        await _invalidate_projects_summary_cache(user_id)
    return deleted
