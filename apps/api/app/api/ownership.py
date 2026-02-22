"""Shared project ownership checks for API routes."""

from fastapi import HTTPException

from app.core.db import get_db_pool
from app.repositories.project import fetch_project


async def require_owned_project(project_id: str, user_id: str) -> dict:
    """
    Return the owned project row or raise 404.

    Centralizes the common router pattern so ownership checks remain consistent.
    """
    pool = await get_db_pool()
    project = await fetch_project(pool, project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

