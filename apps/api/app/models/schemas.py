"""Pydantic request/response schemas for all API routes.

Centralised here so that schemas can be shared across routes and services
without circular imports. Route files import from here — never define
BaseModel subclasses directly in route files.
"""

import uuid as _uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _validate_uuid(value: str, field_name: str) -> str:
    """Reject malformed UUIDs early so they never reach the DB."""
    try:
        _uuid.UUID(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid UUID")
    return value


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    project_id: str
    session_id: str | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be whitespace-only")
        return v

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        return _validate_uuid(v, "project_id")


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    refresh_strategy: str = Field(default="once")
    tiktok_enabled: bool = True
    instagram_enabled: bool = True
    openalex_enabled: bool = True


class ProjectResponse(BaseModel):
    id: str
    title: str
    description: str
    refresh_strategy: str
    structured_intent: dict = Field(default_factory=dict)
    kb_chunk_count: int = 0
    tiktok_enabled: bool = True
    instagram_enabled: bool = True
    openalex_enabled: bool = True
    last_refreshed_at: str | None = None
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    upload_id: str
    status: str
    filename: str
    project_id: str


class KBStatusResponse(BaseModel):
    project_id: str
    kb_chunk_count: int
    last_refreshed_at: str | None
    status: str


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class DiscoverItem(BaseModel):
    platform: Literal["tiktok", "instagram"]
    video_id: str
    url: str | None = None
    cover_url: str | None = None
    caption: str
    author: str
    likes: int = 0
    views: int = 0


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    """Structured intent extracted by the IntentAgent from a user query.

    ``is_research_query`` gates whether the PatentAgent / OpenAlex pipeline
    is invoked downstream.  It must default to ``False`` so that trend /
    social queries never trigger the academic-paper retrieval path.
    """

    domain: str
    query_type: Literal["trend", "research", "patent", "social", "general"]
    entities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    is_research_query: bool = False


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    user_id: str
    tier: str = "free"
