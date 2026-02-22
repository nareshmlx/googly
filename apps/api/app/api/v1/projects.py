"""Projects router — HTTP layer only.

Rule: No business logic here. Validate input, call ProjectService, return response.
user_id comes from X-User-ID header (set by APIM in production, passed directly in dev).
arq_pool is injected via FastAPI dependency so it can be mocked in tests.
"""

import json
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.api.ownership import require_owned_project
from app.core.arq import get_arq_pool
from app.core.auth import get_current_user, verify_internal_token
from app.core.constants import ProjectRefresh
from app.models.schemas import DiscoverItem, ProjectCreate, ProjectResponse
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
    return ProjectResponse(
        id=p["id"],
        title=p["title"],
        description=p["description"],
        refresh_strategy=p["refresh_strategy"],
        structured_intent=(
            p["structured_intent"]
            if isinstance(p.get("structured_intent"), dict)
            else json.loads(p["structured_intent"])
            if isinstance(p.get("structured_intent"), str)
            else {}
        ),
        kb_chunk_count=p.get("kb_chunk_count") or 0,
        tiktok_enabled=p.get("tiktok_enabled", True),
        instagram_enabled=p.get("instagram_enabled", True),
        openalex_enabled=p.get("openalex_enabled", True),
        last_refreshed_at=p["last_refreshed_at"].isoformat()
        if p.get("last_refreshed_at")
        else None,
        created_at=p["created_at"].isoformat() if p.get("created_at") else None,
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
            openalex_enabled=body.openalex_enabled,
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
    Return the Discover feed for a project (TikTok + Instagram social chunks).

    Returns 404 if the project does not exist or is not owned by the requesting user.
    Returns an empty list when no social content has been ingested yet — this is
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
        metadata: dict = (
            row["metadata"]
            if isinstance(row.get("metadata"), dict)
            else json.loads(row["metadata"])
            if isinstance(row.get("metadata"), str)
            else {}
        )
        platform: Literal["tiktok", "instagram"] = (
            "tiktok" if row["source"] == "social_tiktok" else "instagram"
        )
        items.append(
            DiscoverItem(
                platform=platform,
                video_id=row["source_id"],
                url=metadata.get("video_url") or metadata.get("url") or None,
                cover_url=metadata.get("cover_url") or None,
                caption=row["content"][:500],
                author=row["title"],
                likes=int(metadata.get("likes") or metadata.get("like_count") or 0),
                views=int(metadata.get("views") or metadata.get("view_count") or 0),
            )
        )

    return items
