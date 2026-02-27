"""SQLAlchemy ORM models — match production schema from PROJECT_OVERVIEW.md §7.

Used only by Alembic autogenerate. Runtime DB access goes through asyncpg directly.
"""

from datetime import UTC, datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    clerk_id = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=False)
    tier = Column(String(50), nullable=False, server_default="free")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    # user_id is stored as TEXT — Clerk IDs are "user_2abc..." not UUIDs.
    # The FK to users.id was dropped in migration 002.
    user_id = Column(Text, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    structured_intent = Column(JSONB, nullable=False, server_default="{}")
    refresh_strategy = Column(String(50), nullable=False, server_default="once")
    last_refreshed_at = Column(DateTime(timezone=True), nullable=True)
    kb_chunk_count = Column(Integer, nullable=False, server_default="0")
    tiktok_enabled = Column(Boolean, nullable=False, server_default="true")
    instagram_enabled = Column(Boolean, nullable=False, server_default="true")
    youtube_enabled = Column(Boolean, nullable=False, server_default="true")
    reddit_enabled = Column(Boolean, nullable=False, server_default="true")
    x_enabled = Column(Boolean, nullable=False, server_default="true")
    papers_enabled = Column(Boolean, nullable=False, server_default="true")
    perigon_enabled = Column(Boolean, nullable=False, server_default="true")
    tavily_enabled = Column(Boolean, nullable=False, server_default="true")
    exa_enabled = Column(Boolean, nullable=False, server_default="true")
    metadata = Column(JSONB, nullable=False, server_default="{}")
    patents_enabled = Column(Boolean, nullable=False, server_default="true")
    intent_embedding = Column(Vector(dim=1536), nullable=True)  # type: ignore[var-annotated]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    user = relationship("User", back_populates="projects", foreign_keys=[user_id], viewonly=True)
    knowledge_chunks = relationship(
        "KnowledgeChunk", back_populates="project", cascade="all, delete-orphan"
    )
    chat_messages = relationship(
        "ChatMessage", back_populates="project", cascade="all, delete-orphan"
    )
    source_assets = relationship(
        "KnowledgeSourceAsset", back_populates="project", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "refresh_strategy IN ('once', 'daily', 'weekly', 'on_demand')",
            name="ck_projects_refresh_strategy",
        ),
        Index("ix_projects_user_id", "user_id"),
    )


class KnowledgeChunk(Base):
    """
    Stores one text chunk + its vector embedding per project.

    `user_id` is denormalized here (also on projects) for fast cross-project queries
    without an extra join. `source` tracks where the chunk came from so we can
    deduplicate by (project_id, source, source_id) and expire news/social chunks.
    `embedding` is declared as Text here so SQLAlchemy can manage the model —
    the actual vector(1536) DDL is handled in the migration via raw ALTER TABLE.
    """

    __tablename__ = "knowledge_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    # user_id denormalized as TEXT — Clerk IDs are not UUIDs (migration 002)
    user_id = Column(Text, nullable=False)
    source = Column(String(50), nullable=False)  # upload|news|paper|patent|social|search
    source_id = Column(Text, nullable=True)
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    # embedding stored as vector(1536) in DB; ORM uses Text to avoid pgvector type registration
    metadata_ = Column("metadata", JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    project = relationship("Project", back_populates="knowledge_chunks")

    __table_args__ = (
        UniqueConstraint("project_id", "source", "source_id", name="uq_kc_project_source"),
        Index("ix_kc_project_id", "project_id"),
        Index("ix_kc_user_id", "user_id"),
    )


class ChatMessage(Base):
    """Durable chat history rows scoped by project + user + session."""

    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(Text, nullable=False)
    session_id = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    project = relationship("Project", back_populates="chat_messages")

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_chat_messages_role",
        ),
        Index("ix_chat_messages_project_session_created", "project_id", "session_id", "created_at"),
        Index("ix_chat_messages_user_project_created", "user_id", "project_id", "created_at"),
    )


class KnowledgeSourceAsset(Base):
    """Tracks fulltext asset fetch/extract lifecycle for paper/patent enrichment."""

    __tablename__ = "knowledge_source_assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(Text, nullable=False)
    source = Column(String(50), nullable=False)
    source_id = Column(Text, nullable=False)
    title = Column(Text, nullable=False, server_default="")
    source_url = Column(Text, nullable=False)
    resolved_url = Column(Text, nullable=False)
    canonical_url = Column(Text, nullable=False)
    asset_type = Column(String(32), nullable=False, server_default="pdf")
    mime_type = Column(String(255), nullable=False, server_default="")
    blob_path = Column(Text, nullable=False, server_default="")
    checksum_sha256 = Column(String(64), nullable=False, server_default="")
    byte_size = Column(BigInteger, nullable=False, server_default="0")
    fetch_status = Column(String(32), nullable=False, server_default="pending")
    extract_status = Column(String(32), nullable=False, server_default="pending")
    error_code = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    attempt_count = Column(Integer, nullable=False, server_default="0")
    next_attempt_at = Column(DateTime(timezone=True), nullable=True)
    last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    extracted_chars = Column(Integer, nullable=False, server_default="0")
    extracted_pages = Column(Integer, nullable=False, server_default="0")
    metadata = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    project = relationship("Project", back_populates="source_assets")

    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "source",
            "source_id",
            "canonical_url",
            name="uq_ksa_project_source_sourceid_canonical",
        ),
        Index("ix_ksa_fetch_status", "fetch_status"),
        Index("ix_ksa_next_attempt_at", "next_attempt_at"),
        Index("ix_ksa_project_source", "project_id", "source"),
    )
