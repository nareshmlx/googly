"""Projects router — HTTP layer only.

Rule: No business logic here. Validate input, call ProjectService, return response.
user_id comes from X-User-ID header (set by APIM in production, passed directly in dev).
arq_pool is injected via FastAPI dependency so it can be mocked in tests.
"""

import json
from datetime import datetime
from typing import Literal, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.api.ownership import require_owned_project
from app.core.arq import get_arq_pool
from app.core.auth import get_current_user, verify_internal_token
from app.core.constants import ProjectRefresh
from app.models.schemas import DiscoverItem, IngestStatusResponse, ProjectCreate, ProjectResponse
from app.services import project as project_service

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_internal_token)])


async def _run_service_call(action: str, detail: str, coro):
    """Translate unexpected service failures to consistent 500 responses."""
    try:
        return await coro
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"projects.{action}.error")
        raise HTTPException(status_code=500, detail=detail)


def _serialize_project(p: dict) -> ProjectResponse:
    """Convert asyncpg row dict to ProjectResponse — handles datetime serialisation."""
    structured_intent_raw = p.get("structured_intent")
    if isinstance(structured_intent_raw, dict):
        structured_intent = structured_intent_raw
    elif isinstance(structured_intent_raw, str):
        try:
            parsed_intent = json.loads(structured_intent_raw)
            structured_intent = parsed_intent if isinstance(parsed_intent, dict) else {}
        except json.JSONDecodeError:
            structured_intent = {}
    else:
        structured_intent = {}

    created_at_raw = p.get("created_at")
    if isinstance(created_at_raw, datetime):
        created_at = created_at_raw.isoformat()
    elif created_at_raw is not None:
        created_at = str(created_at_raw)
    else:
        raise ValueError("project row missing created_at")

    return ProjectResponse(
        id=p["id"],
        title=p["title"],
        description=p["description"],
        refresh_strategy=p["refresh_strategy"],
        structured_intent=structured_intent,
        kb_chunk_count=p.get("kb_chunk_count") or 0,
        tiktok_enabled=p.get("tiktok_enabled", True),
        instagram_enabled=p.get("instagram_enabled", True),
        papers_enabled=p.get("papers_enabled", True),
        patents_enabled=p.get("patents_enabled", True),
        perigon_enabled=p.get("perigon_enabled", True),
        tavily_enabled=p.get("tavily_enabled", True),
        exa_enabled=p.get("exa_enabled", True),
        last_refreshed_at=p["last_refreshed_at"].isoformat()
        if p.get("last_refreshed_at")
        else None,
        created_at=created_at,
    )


def _row_to_discover_item(row: dict) -> DiscoverItem | None:
    """Map a DB row into the unified DiscoverItem response schema.

    Returns None for unsupported source types.
    """
    raw_source = str(row.get("source") or "search")
    source: Literal["tiktok", "instagram", "paper", "patent", "news", "search"] | None = (
        "tiktok"
        if raw_source == "social_tiktok"
        else "instagram"
        if raw_source == "social_instagram"
        else None
    )
    if raw_source in {"paper", "patent", "news", "search"}:
        source = cast(
            Literal["tiktok", "instagram", "paper", "patent", "news", "search"], raw_source
        )
    if source is None:
        logger.warning("projects.discover.unsupported_source", source=raw_source)
        return None

    metadata_raw = row.get("metadata")
    metadata: dict
    if isinstance(metadata_raw, dict):
        metadata = metadata_raw
    elif isinstance(metadata_raw, str):
        try:
            parsed_meta = json.loads(metadata_raw)
            metadata = parsed_meta if isinstance(parsed_meta, dict) else {}
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}

    title = str(
        row.get("title")
        or metadata.get("title")
        or metadata.get("author")
        or metadata.get("byline")
        or "Untitled"
    )

    url_raw = metadata.get("url") or metadata.get("doi") or row.get("url")
    url = str(url_raw) if url_raw else None

    row_created_at = row.get("created_at")
    if isinstance(row_created_at, datetime):
        created_at_str = row_created_at.isoformat()
    elif row_created_at is not None:
        created_at_str = str(row_created_at)
    else:
        created_at_str = None

    published_raw = (
        metadata.get("published_at")
        or metadata.get("pubDate")
        or metadata.get("year")
        or row.get("published_at")
        or created_at_str
    )
    published_at = str(published_raw) if published_raw else None

    summary_raw = row.get("summary") or row.get("content") or metadata.get("summary") or ""

    score_raw = row.get("score")
    try:
        score = float(score_raw) if score_raw is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0

    cover_raw = metadata.get("cover_url") or metadata.get("thumbnail_url")
    cover_url = str(cover_raw) if cover_raw else None

    author_raw = metadata.get("author") or metadata.get("byline")
    author = str(author_raw) if author_raw else None

    return DiscoverItem(
        source=source,
        item_id=str(row.get("item_id") or row.get("id") or row.get("source_id") or ""),
        title=title,
        summary=str(summary_raw)[:500],
        url=url,
        cover_url=cover_url,
        author=author,
        published_at=published_at,
        score=score,
        metadata=metadata,
    )


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: ProjectCreate,
    current_user: dict = Depends(get_current_user),
    arq_pool=Depends(get_arq_pool),
):
    """
    Create a project, extract intent, and enqueue KB ingestion.

    Returns 201 with the full project including extracted structured_intent.
    KB population runs in the background — kb_chunk_count will be 0 initially.
    """
    if body.refresh_strategy not in ProjectRefresh.VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"refresh_strategy must be one of: {', '.join(sorted(ProjectRefresh.VALID_STRATEGIES))}",
        )

    user_id = current_user["user_id"]
    logger.info("projects.create", user_id=user_id, title=body.title)

    project = await _run_service_call(
        action="create",
        detail="Failed to create project",
        coro=project_service.create_project(
            user_id=user_id,
            title=body.title,
            description=body.description,
            refresh_strategy=body.refresh_strategy,
            arq_pool=arq_pool,
            tiktok_enabled=body.tiktok_enabled,
            instagram_enabled=body.instagram_enabled,
            papers_enabled=body.papers_enabled,
            patents_enabled=body.patents_enabled,
            perigon_enabled=body.perigon_enabled,
            tavily_enabled=body.tavily_enabled,
            exa_enabled=body.exa_enabled,
        ),
    )

    return _serialize_project(project)


@router.get("/", response_model=list[ProjectResponse])
async def list_projects(current_user: dict = Depends(get_current_user)):
    """List all projects owned by the current user, newest first."""
    user_id = current_user["user_id"]
    projects = await _run_service_call(
        action="list",
        detail="Failed to list projects",
        coro=project_service.list_projects(user_id),
    )
    return [_serialize_project(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get a single project. Returns 404 if not found or not owned by the user."""
    user_id = current_user["user_id"]
    project = await require_owned_project(project_id, user_id)
    return _serialize_project(project)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a project and all its knowledge chunks. Returns 404 if not found."""
    user_id = current_user["user_id"]
    deleted = await project_service.delete_project(project_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")


@router.get("/{project_id}/discover", response_model=list[DiscoverItem])
async def get_discover_feed(
    project_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Return the Discover feed for a project (all supported source types).

    Returns 404 if the project does not exist or is not owned by the requesting user.
    Returns an empty list when no content has been ingested yet — this is
    not an error.
    """
    user_id = current_user["user_id"]

    await require_owned_project(project_id, user_id)

    rows = await _run_service_call(
        action="discover",
        detail="Failed to fetch discover feed",
        coro=project_service.get_discover_feed(project_id),
    )

    items: list[DiscoverItem] = []
    for row in rows:
        item = _row_to_discover_item(row)
        if item is not None:
            items.append(item)
    return items


@router.get("/{project_id}/ingest-status", response_model=IngestStatusResponse)
async def get_ingest_status(
    project_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return latest ingestion lifecycle status for a project."""
    user_id = current_user["user_id"]
    await require_owned_project(project_id, user_id)
    payload = await _run_service_call(
        action="ingest_status",
        detail="Failed to fetch ingest status",
        coro=project_service.get_project_ingest_status(project_id),
    )
    return IngestStatusResponse(**payload)
