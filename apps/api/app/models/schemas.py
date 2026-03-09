"""Pydantic request/response schemas for all API routes.

Centralised here so that schemas can be shared across routes and services
without circular imports. Route files import from here — never define
BaseModel subclasses directly in route files.
"""

import uuid as _uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


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
    youtube_enabled: bool = True
    reddit_enabled: bool = True
    x_enabled: bool = True
    papers_enabled: bool = True
    patents_enabled: bool = True
    perigon_enabled: bool = True
    tavily_enabled: bool = True
    exa_enabled: bool = True


class ProjectResponse(BaseModel):
    id: str
    title: str
    description: str
    enriched_description: str | None = None
    refresh_strategy: str
    structured_intent: dict = Field(default_factory=dict)
    kb_chunk_count: int = 0
    tiktok_enabled: bool
    instagram_enabled: bool
    youtube_enabled: bool
    reddit_enabled: bool
    x_enabled: bool
    papers_enabled: bool
    patents_enabled: bool
    perigon_enabled: bool
    tavily_enabled: bool
    exa_enabled: bool
    last_refreshed_at: str | None = None
    created_at: str


class ProjectBootstrapRequest(BaseModel):
    upload_ids: list[str] = Field(default_factory=list)


class WizardQAPair(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    answer: str = Field(..., min_length=1, max_length=5000)
    dimension: str | None = Field(default=None, max_length=64)


class WizardEvaluateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    qa_pairs: list[WizardQAPair] = Field(default_factory=list)
    max_questions: int = Field(default=5, ge=1, le=10)


class WizardEvaluateResponse(BaseModel):
    scores: dict = Field(default_factory=dict)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    weakest_dimension: str | None = None
    next_dimension: str | None = None
    next_question: str | None = None
    should_stop: bool = False
    asked_questions: int = Field(default=0, ge=0)
    max_questions: int = Field(default=5, ge=1)


class WizardSynthesizeRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    qa_pairs: list[WizardQAPair] = Field(default_factory=list)
    structured_intent: dict = Field(default_factory=dict)
    source_toggles: dict = Field(default_factory=dict)


class WizardSynthesizeResponse(BaseModel):
    enriched_description: str = Field(default="", max_length=12000)
    domain_focus: str = Field(default="", max_length=255)
    key_entities: list[str] = Field(default_factory=list)
    must_match_terms: list[str] = Field(default_factory=list)
    time_horizon: str = Field(default="", max_length=128)
    target_sources: dict = Field(default_factory=dict)


class WizardCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    qa_pairs: list[WizardQAPair] = Field(default_factory=list)
    refresh_strategy: str = Field(default="once")
    enriched_description: str = Field(default="", max_length=12000)
    domain_focus: str = Field(default="", max_length=255)
    key_entities: list[str] = Field(default_factory=list)
    must_match_terms: list[str] = Field(default_factory=list)
    time_horizon: str = Field(default="", max_length=128)
    target_sources: dict = Field(default_factory=dict)


class ProjectSetupStatusResponse(BaseModel):
    project_id: str
    status: str
    phase: str
    progress_percent: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    updated_at: str | None = None
    error: str | None = None
    upload_ids: list[str] = Field(default_factory=list)
    upload_signature: str = ""
    job_id: str | None = None


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


class IngestStatusResponse(BaseModel):
    project_id: str
    status: str
    message: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    job_id: str | None = None
    total_chunks: int | None = None
    source_counts: dict = Field(default_factory=dict)
    source_diagnostics: dict = Field(default_factory=dict)
    cluster_diagnostics: dict = Field(default_factory=dict)
    fulltext_enqueued: int = 0
    enrichment: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class DiscoverItem(BaseModel):
    source: Literal[
        "tiktok",
        "instagram",
        "youtube",
        "reddit",
        "x",
        "paper",
        "patent",
        "news",
        "search",
    ]
    item_id: str
    title: str
    summary: str
    url: str | None = None
    cover_url: str | None = None
    author: str | None = None
    published_at: str | None = None
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


class KeyFinding(BaseModel):
    text: str


class InsightCard(BaseModel):
    id: str
    topic_label: str
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    trend_signal: Literal["rising", "stable", "declining", "emerging", "unknown"] = "unknown"
    cluster_size: int
    source_doc_count: int = 0
    full_report_status: Literal["pending", "generating", "done", "failed"] = "pending"
    source_type_counts: dict[str, int] = Field(default_factory=dict)


class SourceDocSummary(BaseModel):
    id: str
    title: str
    url: str | None = None
    summary: str | None = None
    source: str
    cover_url: str | None = None
    video_url: str | None = None
    author: str | None = None
    views: int | None = None
    likes: int | None = None
    published_at: str | None = None


class InsightDetail(BaseModel):
    id: str
    topic_label: str
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    trend_signal: Literal["rising", "stable", "declining", "emerging", "unknown"] = "unknown"
    contradictions: str | None = None
    cluster_size: int
    source_doc_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    full_report: str | None = None
    full_report_status: Literal["pending", "generating", "done", "failed"] = "pending"
    source_type_counts: dict[str, int] = Field(default_factory=dict)
    source_docs: list[SourceDocSummary] = Field(default_factory=list)


class ClusterExtraction(BaseModel):
    topic_label: str = Field(..., min_length=1, max_length=200)
    executive_summary: str = Field(..., min_length=1, max_length=300)
    key_findings: list[str] = Field(default_factory=list, max_length=settings.CLUSTER_MAX_KEY_FINDINGS)
    trend_signal: Literal["rising", "stable", "declining", "emerging", "unknown"]
    contradictions: str | None = None


class FollowupRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be whitespace-only")
        return v


class FollowupMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    context_source: Literal["cluster", "cluster_docs_expanded"]
    created_at: str


class InsightRefreshResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    """Structured intent extracted by the IntentAgent from a user query.

    ``is_research_query`` gates whether the PatentAgent / OpenAlex pipeline
    is invoked downstream.  It must default to ``False`` so that trend /
    social queries never trigger the academic-paper retrieval path.

    ``target_domain`` carries an optional domain filter extracted when the user
    explicitly names a specific website or publication (e.g. "techcrunch.com").
    None means no domain restriction — all web sources are searched.
    """

    domain: str
    query_type: Literal["trend", "research", "patent", "social", "general"]
    entities: list[str] = Field(default_factory=list)
    must_match_terms: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    query_specificity: Literal["specific", "broad"] = "broad"
    confidence: float = Field(ge=0.0, le=1.0)
    is_research_query: bool = False
    target_domain: str | None = None


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    user_id: str
    tier: str = "free"
