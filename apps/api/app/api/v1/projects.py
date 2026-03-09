"""Projects router — HTTP layer only.

Rule: No business logic here. Validate input, call ProjectService, return response.
user_id comes from X-User-ID header (set by APIM in production, passed directly in dev).
arq_pool is injected via FastAPI dependency so it can be mocked in tests.
"""

from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.api.ownership import require_owned_project
from app.core.arq import get_arq_pool
from app.core.auth import get_current_user, verify_internal_token
from app.core.constants import ProjectRefresh
from app.core.utils import parse_metadata
from app.models.schemas import (
    DiscoverItem,
    IngestStatusResponse,
    ProjectBootstrapRequest,
    ProjectCreate,
    ProjectResponse,
    ProjectSetupStatusResponse,
    WizardCreateRequest,
    WizardEvaluateRequest,
    WizardEvaluateResponse,
    WizardSynthesizeRequest,
    WizardSynthesizeResponse,
)
from app.services import project as project_service
from app.services import project_discover
from app.services import project_wizard as project_wizard_service

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
    structured_intent = parse_metadata(p.get("structured_intent"))

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
        enriched_description=p.get("enriched_description"),
        refresh_strategy=p["refresh_strategy"],
        structured_intent=structured_intent,
        kb_chunk_count=p.get("kb_chunk_count") or 0,
        tiktok_enabled=p.get("tiktok_enabled", True),
        instagram_enabled=p.get("instagram_enabled", True),
        youtube_enabled=p.get("youtube_enabled", True),
        reddit_enabled=p.get("reddit_enabled", True),
        x_enabled=p.get("x_enabled", True),
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


def _normalize_target_sources(raw: dict | None) -> dict[str, bool]:
    defaults = {
        "tiktok": True,
        "instagram": True,
        "youtube": True,
        "reddit": True,
        "x": True,
        "papers": True,
        "patents": True,
        "news": True,
        "web_tavily": True,
        "web_exa": True,
    }
    payload = raw or {}
    if "web" in payload:
        defaults["web_tavily"] = bool(payload.get("web"))
        defaults["web_exa"] = bool(payload.get("web"))
    for key in defaults:
        if key in payload:
            defaults[key] = bool(payload[key])
    return defaults


def _qa_payload(qa_pairs: list) -> list[dict]:
    return [
        {
            "question": str(item.question),
            "answer": str(item.answer),
            "dimension": str(item.dimension or ""),
        }
        for item in qa_pairs
    ]


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
            youtube_enabled=body.youtube_enabled,
            reddit_enabled=body.reddit_enabled,
            x_enabled=body.x_enabled,
            papers_enabled=body.papers_enabled,
            patents_enabled=body.patents_enabled,
            perigon_enabled=body.perigon_enabled,
            tavily_enabled=body.tavily_enabled,
            exa_enabled=body.exa_enabled,
        ),
    )

    return _serialize_project(project)


@router.post("/wizard/evaluate", response_model=WizardEvaluateResponse)
async def wizard_evaluate(
    body: WizardEvaluateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Evaluate wizard sufficiency and generate the next focused question."""
    _ = current_user["user_id"]
    payload = await _run_service_call(
        action="wizard_evaluate",
        detail="Failed to evaluate project wizard context",
        coro=project_wizard_service.evaluate_project_sufficiency(
            title=body.title,
            description=body.description,
            qa_pairs=_qa_payload(body.qa_pairs),
            max_questions=body.max_questions,
        ),
    )
    return WizardEvaluateResponse(**payload)


@router.post("/wizard/synthesize", response_model=WizardSynthesizeResponse)
async def wizard_synthesize(
    body: WizardSynthesizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Synthesize wizard context into phase-2 review fields."""
    _ = current_user["user_id"]
    payload = await _run_service_call(
        action="wizard_synthesize",
        detail="Failed to synthesize wizard review payload",
        coro=project_wizard_service.synthesize_wizard_review(
            title=body.title,
            description=body.description,
            qa_pairs=_qa_payload(body.qa_pairs),
            structured_intent=parse_metadata(body.structured_intent),
            source_toggles=body.source_toggles,
        ),
    )
    return WizardSynthesizeResponse(**payload)


@router.post("/wizard/create", response_model=ProjectResponse, status_code=201)
async def wizard_create_project(
    body: WizardCreateRequest,
    current_user: dict = Depends(get_current_user),
    arq_pool=Depends(get_arq_pool),
):
    """
    Create a project using two-phase wizard context and manual overrides.

    Phase-1 context is used to produce an enriched description, then Phase-2
    overrides are merged into structured_intent values while preserving schema.
    """
    if body.refresh_strategy not in ProjectRefresh.VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"refresh_strategy must be one of: {', '.join(sorted(ProjectRefresh.VALID_STRATEGIES))}",
        )

    source_toggles = _normalize_target_sources(body.target_sources)

    user_id = current_user["user_id"]
    created_project_id = ""
    try:
        project = await _run_service_call(
            action="wizard_create",
            detail="Failed to create wizard project",
            coro=project_service.create_project(
                user_id=user_id,
                title=body.title,
                description=body.description,
                refresh_strategy=body.refresh_strategy,
                arq_pool=arq_pool,
                tiktok_enabled=source_toggles["tiktok"],
                instagram_enabled=source_toggles["instagram"],
                youtube_enabled=source_toggles["youtube"],
                reddit_enabled=source_toggles["reddit"],
                x_enabled=source_toggles["x"],
                papers_enabled=source_toggles["papers"],
                patents_enabled=source_toggles["patents"],
                perigon_enabled=source_toggles["news"],
                tavily_enabled=source_toggles["web_tavily"],
                exa_enabled=source_toggles["web_exa"],
                enqueue_ingestion=False,
                skip_description_expansion=True,
                intent_seed_text=project_wizard_service.build_intent_seed_text(
                    description=body.description,
                    domain_focus=body.domain_focus,
                    key_entities=body.key_entities,
                    must_match_terms=body.must_match_terms,
                ),
            ),
        )
        created_project_id = str(project.get("id") or "")

        enriched_description = str(body.enriched_description or "").strip() or str(
            body.description or ""
        ).strip()

        updated = await _run_service_call(
            action="wizard_merge",
            detail="Failed to merge wizard overrides into intent",
            coro=project_service.apply_wizard_overrides(
                project_id=created_project_id,
                enriched_description=enriched_description,
                overrides={
                    "domain_focus": body.domain_focus,
                    "key_entities": body.key_entities,
                    "must_match_terms": body.must_match_terms,
                    "time_horizon": body.time_horizon,
                    "target_sources": source_toggles,
                },
            ),
        )
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to finalize wizard project")
    except HTTPException:
        if created_project_id:
            deleted = await project_service.delete_project(created_project_id, user_id)
            if not deleted:
                logger.warning(
                    "projects.wizard_create.rollback_failed",
                    project_id=created_project_id,
                    user_id=user_id,
                )
        raise
    except Exception:
        if created_project_id:
            deleted = await project_service.delete_project(created_project_id, user_id)
            if not deleted:
                logger.warning(
                    "projects.wizard_create.rollback_failed",
                    project_id=created_project_id,
                    user_id=user_id,
                )
        logger.exception("projects.wizard_create.unexpected")
        raise HTTPException(status_code=500, detail="Failed to finalize wizard project")

    await project_service.enqueue_project_ingestion(created_project_id, arq_pool)
    return _serialize_project(updated)


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
        item = project_discover.row_to_discover_item(row)
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


@router.post("/{project_id}/bootstrap", response_model=ProjectSetupStatusResponse)
async def bootstrap_project(
    project_id: str,
    body: ProjectBootstrapRequest,
    current_user: dict = Depends(get_current_user),
    arq_pool=Depends(get_arq_pool),
):
    """Enqueue async project setup/bootstrap with idempotent active-run handling."""
    user_id = current_user["user_id"]
    await require_owned_project(project_id, user_id)
    payload = await _run_service_call(
        action="bootstrap",
        detail="Failed to bootstrap project setup",
        coro=project_service.enqueue_project_bootstrap(
            project_id=project_id,
            user_id=user_id,
            upload_ids=body.upload_ids,
            arq_pool=arq_pool,
        ),
    )
    return ProjectSetupStatusResponse(**payload)
