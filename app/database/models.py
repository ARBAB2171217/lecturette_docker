"""
SQLAlchemy ORM models:
    - Lecturette       -> final generated lecturettes
    - ResearchCache    -> cached research + embeddings (pgvector)
    - SearchLog        -> telemetry for every incoming request
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config.settings import settings
from app.database.connection import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Lecturette(Base):
    __tablename__ = "lecturettes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    topic: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    lecturette: Mapped[str] = mapped_column(Text, nullable=False)
    speaking_duration: Mapped[str] = mapped_column(
        String(64), default="Approx 3 Minutes"
    )
    research_cache_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_cache.id"), nullable=True
    )
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="database")  # database/web
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    research_cache: Mapped["ResearchCache | None"] = relationship(
        back_populates="lecturettes"
    )


class ResearchCache(Base):
    __tablename__ = "research_cache"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    topic: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=False
    )

    research_summary: Mapped[str] = mapped_column(Text, nullable=False)
    raw_research: Mapped[str] = mapped_column(Text, nullable=True)
    compressed_research: Mapped[dict] = mapped_column(JSONB, nullable=False)
    sources: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    web_search_performed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    lecturettes: Mapped[list["Lecturette"]] = relationship(
        back_populates="research_cache"
    )

    __table_args__ = (
        Index(
            "ix_research_cache_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class SearchLog(Base):
    __tablename__ = "search_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    used_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    web_search_performed: Mapped[bool] = mapped_column(Boolean, default=False)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
