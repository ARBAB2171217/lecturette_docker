"""
API Routes — /generate-lecturette and supporting endpoints.

Orchestrates the full pipeline:
    Parser -> Retrieval (cache-first) -> Research Agent -> Writer Agent -> Persist
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.research_agent import ResearchAgent
from app.agents.writer_agent import WriterAgent
from app.database.connection import get_db
from app.database.models import Lecturette, SearchLog
from app.schemas.lecturette_schema import LecturetteRequest, LecturetteResponse
from app.services.parser_service import ParserService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Lecturette"])


@router.post("/generate-lecturette", response_model=LecturetteResponse)
async def generate_lecturette(
    payload: LecturetteRequest,
    db: AsyncSession = Depends(get_db),
) -> LecturetteResponse:
    start = time.perf_counter()

    try:
        # 1. Parse topic -> keywords, category, normalized query
        parsed = await ParserService.parse_topic(payload.topic)
        main_topic = parsed["main_topic"]
        category = parsed["category"]
        keywords = parsed.get("keywords", [])
        normalized_query = parsed["normalized_query"]

        # 2. Retrieval (embedding + vector search + cache-first/Google Search decision)
        retrieval = await RetrievalService.retrieve(
            db, main_topic, category, keywords, normalized_query
        )

        # 3. Research Agent -> structured notes + outline
        structured = await ResearchAgent.build_structure(
            topic=main_topic,
            category=category,
            research_summary=retrieval.research_entry.research_summary,
            compressed_research=retrieval.research_entry.compressed_research,
        )

        # 4. Writer Agent -> final markdown lecturette
        lecturette_md = await WriterAgent.write_lecturette(structured)

        # 5. Persist final lecturette
        lecturette_row = Lecturette(
            topic=main_topic,
            category=category,
            lecturette=lecturette_md,
            research_cache_id=retrieval.research_entry.id,
            similarity_score=retrieval.similarity_score,
            source="database" if retrieval.cache_hit else "web",
        )
        db.add(lecturette_row)

        # 6. Log this request
        elapsed_ms = (time.perf_counter() - start) * 1000
        db.add(
            SearchLog(
                query=payload.topic,
                used_cache=retrieval.cache_hit,
                web_search_performed=retrieval.web_search_performed,
                similarity_score=retrieval.similarity_score,
                latency_ms=elapsed_ms,
            )
        )

        await db.commit()

        return LecturetteResponse(
            topic=main_topic,
            category=category,
            lecturette=lecturette_md,
            source="database" if retrieval.cache_hit else "web",
            similarity_score=round(retrieval.similarity_score, 4),
            saved=True,
            cache_hit=retrieval.cache_hit,
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error generating lecturette for topic '%s'", payload.topic)
        raise HTTPException(
            status_code=500, detail="Internal error generating lecturette."
        ) from exc
