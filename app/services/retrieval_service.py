"""
Retrieval Service — implements the cache-first, retrieval-first strategy:

    1. Embed the parsed topic.
    2. Vector-search research_cache for top-K similar topics.
    3. If best match similarity >= SIMILARITY_CACHE_HIT_THRESHOLD -> reuse it,
       skip Google Search entirely.
    4. Else -> run Google Search, build fresh research, and either:
         - update the existing near-duplicate row (similarity > DUPLICATE_THRESHOLD), or
         - insert a brand-new research_cache row.

This is the only module that decides "do we hit the web or not", keeping
that decision out of the agents themselves.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.database.models import ResearchCache
from app.database.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.google_search_service import google_search_service

logger = logging.getLogger(__name__)


class RetrievalResult:
    def __init__(
        self,
        cache_hit: bool,
        web_search_performed: bool,
        similarity_score: float,
        research_entry: ResearchCache,
        related_topics: list[tuple[ResearchCache, float]],
    ) -> None:
        self.cache_hit = cache_hit
        self.web_search_performed = web_search_performed
        self.similarity_score = similarity_score
        self.research_entry = research_entry
        self.related_topics = related_topics


class RetrievalService:
    @staticmethod
    async def retrieve(
        db: AsyncSession,
        main_topic: str,
        category: str,
        keywords: list[str],
        normalized_query: str,
    ) -> RetrievalResult:
        embedding = await EmbeddingService.embed_topic(normalized_query, keywords)

        similar = await VectorStore.find_similar(
            db, embedding, top_k=settings.TOP_K_SIMILAR_TOPICS
        )

        best_entry, best_score = (similar[0] if similar else (None, 0.0))

        # ---------------- CACHE HIT ----------------
        if best_entry is not None and best_score >= settings.SIMILARITY_CACHE_HIT_THRESHOLD:
            logger.info(
                "Cache HIT for topic='%s' (score=%.3f, matched='%s')",
                main_topic, best_score, best_entry.topic,
            )
            return RetrievalResult(
                cache_hit=True,
                web_search_performed=False,
                similarity_score=best_score,
                research_entry=best_entry,
                related_topics=similar,
            )

        # ---------------- CACHE MISS -> Google Search ----------------
        logger.info(
            "Cache MISS for topic='%s' (best_score=%.3f); querying Google Search.",
            main_topic, best_score,
        )

        gap_knowledge = RetrievalService._merge_related_knowledge(similar)

        queries = google_search_service.build_research_queries(main_topic, keywords)
        web_results = []
        all_sources: list[str] = []
        combined_answer_parts: list[str] = []

        for q in queries:
            result = await google_search_service.search(q)
            if result["answer"]:
                combined_answer_parts.append(result["answer"])
            web_results.extend(result["results"])
            all_sources.extend(result["sources"])

        raw_research = RetrievalService._format_raw_research(
            main_topic, gap_knowledge, web_results, combined_answer_parts
        )
        research_summary = (
            " ".join(combined_answer_parts)[:1500]
            if combined_answer_parts
            else raw_research[:1500]
        )
        compressed_research = RetrievalService._compress(
            main_topic, web_results, combined_answer_parts, gap_knowledge
        )

        # Near-duplicate -> update in place instead of inserting a new row
        if best_entry is not None and best_score > settings.SIMILARITY_DUPLICATE_THRESHOLD:
            updated = await VectorStore.update_existing(
                db,
                best_entry,
                research_summary=research_summary,
                compressed_research=compressed_research,
                raw_research=raw_research,
                sources=list(set(all_sources)),
            )
            return RetrievalResult(
                cache_hit=False,
                web_search_performed=True,
                similarity_score=best_score,
                research_entry=updated,
                related_topics=similar,
            )

        new_entry = await VectorStore.create(
            db,
            topic=main_topic,
            category=category,
            keywords=keywords,
            embedding=embedding,
            research_summary=research_summary,
            compressed_research=compressed_research,
            raw_research=raw_research,
            sources=list(set(all_sources)),
            web_search_performed=True,
        )
        return RetrievalResult(
            cache_hit=False,
            web_search_performed=True,
            similarity_score=best_score,
            research_entry=new_entry,
            related_topics=similar,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_related_knowledge(
        similar: list[tuple[ResearchCache, float]]
    ) -> str:
        """Pull reusable snippets from related cached topics to fill gaps
        before resorting to a full web search."""
        if not similar:
            return ""
        parts = []
        for entry, score in similar:
            if score < 0.4:
                continue
            parts.append(f"[{entry.topic} | sim={score:.2f}] {entry.research_summary}")
        return "\n".join(parts[:3])

    @staticmethod
    def _format_raw_research(
        main_topic: str,
        gap_knowledge: str,
        web_results: list[dict],
        answers: list[str],
    ) -> str:
        sections = [f"TOPIC: {main_topic}"]
        if gap_knowledge:
            sections.append(f"REUSED KNOWLEDGE FROM RELATED CACHED TOPICS:\n{gap_knowledge}")
        if answers:
            sections.append("GOOGLE SEARCH SYNTHESIZED ANSWERS:\n" + "\n---\n".join(answers))
        for r in web_results[:8]:
            sections.append(
                f"SOURCE: {r.get('url', '')}\nTITLE: {r.get('title', '')}\n"
                f"CONTENT: {r.get('content', '')[:600]}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _compress(
        main_topic: str,
        web_results: list[dict],
        answers: list[str],
        gap_knowledge: str,
    ) -> dict:
        """
        Produces the compressed_research JSON stored in the DB and handed
        to Agent 2. Keeps only facts/stats/arguments/developments — not
        full article text — to minimize tokens sent downstream.
        """
        facts = []
        for r in web_results[:6]:
            content = r.get("content", "")
            if content:
                facts.append(content[:300])

        return {
            "topic": main_topic,
            "key_arguments": answers[:3],
            "facts_and_developments": facts,
            "reused_related_knowledge": gap_knowledge[:1000] if gap_knowledge else "",
        }
