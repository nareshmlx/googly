"""Shared project ownership checks for API routes."""

from fastapi import HTTPException

from app.services import project as project_service


async def require_owned_project(project_id: str, user_id: str) -> dict:
    """
    Return the owned project row or raise 404.

    Centralizes the common router pattern so ownership checks remain consistent.
    """
    project = await project_service.get_project(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
