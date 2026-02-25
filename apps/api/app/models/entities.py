"""SQLAlchemy ORM models — match production schema from PROJECT_OVERVIEW.md §7.

Used only by Alembic autogenerate. Runtime DB access goes through asyncpg directly.
"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
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
