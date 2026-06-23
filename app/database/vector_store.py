"""
Thin data-access layer around pgvector cosine similarity search.
Keeps raw SQL/ORM query logic out of the services & agents.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.database.models import ResearchCache

logger = logging.getLogger(__name__)


class VectorStore:
    """CRUD + similarity-search operations on the research_cache table."""

    @staticmethod
    async def find_similar(
        db: AsyncSession,
        embedding: list[float],
        top_k: int = settings.TOP_K_SIMILAR_TOPICS,
    ) -> list[tuple[ResearchCache, float]]:
        """
        Return up to `top_k` (ResearchCache, similarity_score) pairs,
        ordered by similarity descending. similarity = 1 - cosine_distance.
        """
        distance_col = ResearchCache.embedding.cosine_distance(embedding)
        stmt = (
            select(ResearchCache, distance_col.label("distance"))
            .order_by(distance_col)
            .limit(top_k)
        )
        result = await db.execute(stmt)
        rows = result.all()
        return [(row[0], 1.0 - row[1]) for row in rows]

    @staticmethod
    async def best_match(
        db: AsyncSession, embedding: list[float]
    ) -> tuple[ResearchCache, float] | None:
        matches = await VectorStore.find_similar(db, embedding, top_k=1)
        return matches[0] if matches else None

    @staticmethod
    async def create(
        db: AsyncSession,
        topic: str,
        category: str,
        keywords: list[str],
        embedding: list[float],
        research_summary: str,
        compressed_research: dict,
        raw_research: str = "",
        sources: list[str] | None = None,
        web_search_performed: bool = False,
    ) -> ResearchCache:
        entry = ResearchCache(
            topic=topic,
            category=category,
            keywords=keywords,
            embedding=embedding,
            research_summary=research_summary,
            raw_research=raw_research,
            compressed_research=compressed_research,
            sources=sources or [],
            web_search_performed=web_search_performed,
        )
        db.add(entry)
        await db.flush()
        return entry

    @staticmethod
    async def update_existing(
        db: AsyncSession,
        entry: ResearchCache,
        research_summary: str,
        compressed_research: dict,
        raw_research: str = "",
        sources: list[str] | None = None,
    ) -> ResearchCache:
        entry.research_summary = research_summary
        entry.compressed_research = compressed_research
        if raw_research:
            entry.raw_research = raw_research
        if sources:
            entry.sources = list(set(entry.sources or []) | set(sources))
        entry.updated_at = datetime.now(timezone.utc)
        await db.flush()
        return entry

    @staticmethod
    async def get_by_id(
        db: AsyncSession, research_id: uuid.UUID
    ) -> ResearchCache | None:
        return await db.get(ResearchCache, research_id)
