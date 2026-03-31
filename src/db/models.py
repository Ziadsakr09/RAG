"""
Phase 3 – Database Models
===========================
SQLAlchemy 2.x async models for queries, responses, conversations, and analytics.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from configs.settings import settings
from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(AsyncAttrs, DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class Conversation(Base):
    """Groups related Q&A turns into a conversation session."""
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)

    queries: Mapped[list["QueryRecord"]] = relationship("QueryRecord", back_populates="conversation")


class QueryRecord(Base):
    """Stores every user query with processing metadata."""
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=True)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    normalised_query: Mapped[str] = mapped_column(Text, nullable=False)
    expanded_query: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en")
    is_followup: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped[Optional[Conversation]] = relationship("Conversation", back_populates="queries")
    response: Mapped[Optional["ResponseRecord"]] = relationship("ResponseRecord", back_populates="query", uselist=False)
    analytics: Mapped[Optional["AnalyticsRecord"]] = relationship("AnalyticsRecord", back_populates="query", uselist=False)


class ResponseRecord(Base):
    """Stores generated answers with source tracking."""
    __tablename__ = "responses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id: Mapped[str] = mapped_column(String(36), ForeignKey("queries.id"), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    is_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    model_used: Mapped[str] = mapped_column(String(100), default="")
    sources: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)        # list of doc_ids
    retrieval_scores: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    query: Mapped[QueryRecord] = relationship("QueryRecord", back_populates="response")


class AnalyticsRecord(Base):
    """Per-query performance analytics."""
    __tablename__ = "analytics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id: Mapped[str] = mapped_column(String(36), ForeignKey("queries.id"), nullable=False)
    embedding_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    generation_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    num_results_retrieved: Mapped[int] = mapped_column(Integer, default=0)
    top_retrieval_score: Mapped[float] = mapped_column(Float, default=0.0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    query: Mapped[QueryRecord] = relationship("QueryRecord", back_populates="analytics")


# ─────────────────────────────────────────────────────────────────────────────
# Engine & session factory
# ─────────────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    future=True,
)

async_session_factory = async_sessionmaker(
    engine, expire_on_commit=False
)


async def init_db() -> None:
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised.")


async def get_session():
    """FastAPI dependency: yields an async DB session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
