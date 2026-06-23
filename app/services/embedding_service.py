"""
Embedding Service — generates pgvector-compatible embeddings for topics
and research text using Gemini's gemini-embedding-2 model.
"""

import logging

from app.services.gemini_client import gemini_client

logger = logging.getLogger(__name__)


class EmbeddingService:
    @staticmethod
    async def embed(text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        return await gemini_client.embed_text(text.strip())

    @staticmethod
    async def embed_topic(normalized_query: str, keywords: list[str]) -> list[float]:
        """
        Embeds a combined representation of topic + keywords so the
        similarity search captures both the phrasing and the entities.
        """
        combined = normalized_query
        if keywords:
            combined = f"{normalized_query} | keywords: {', '.join(keywords)}"
        return await EmbeddingService.embed(combined)
