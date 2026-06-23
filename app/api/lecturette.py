"""
API Routes — /generate-lecturette and supporting endpoints.

Orchestrates the full pipeline:
    Parser -> Retrieval (cache-first) -> Research Agent -> Writer Agent -> Persist
"""

import html
import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.research_agent import ResearchAgent
from app.agents.writer_agent import WriterAgent
from app.database.connection import db_session_ctx, get_db
from app.database.models import Lecturette, SearchLog
from app.schemas.lecturette_schema import LecturetteRequest, LecturetteResponse
from app.services.parser_service import ParserService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Lecturette"])


def _markdown_to_html(markdown_text: str) -> str:
    escaped = html.escape(markdown_text).replace("\r\n", "\n")
    escaped = re.sub(r"(?m)^# (.+)$", r"<h1>\1</h1>", escaped)
    escaped = re.sub(r"(?m)^## (.+)$", r"<h2>\1</h2>", escaped)
    escaped = re.sub(r"(?m)^### (.+)$", r"<h3>\1</h3>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?m)^- (.+)$", r"<li>\1</li>", escaped)

    # Wrap adjacent list items in a single <ul>
    escaped = re.sub(
        r"(<li>.*?</li>)(?:\n<li>.*?</li>)*",
        lambda m: "<ul>" + m.group(0).replace("\n", "") + "</ul>",
        escaped,
        flags=re.DOTALL,
    )

    lines = escaped.split("\n")
    result_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<h") or stripped.startswith("<ul>") or stripped.startswith("</ul>") or stripped.startswith("<li>"):
            result_lines.append(line)
        else:
            result_lines.append(f"<p>{line}</p>")

    return "\n".join(result_lines)


def _render_html_page(response: LecturetteResponse) -> str:
    lecturette_html = _markdown_to_html(response.lecturette)
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"UTF-8\">
      <title>Lecturette - {html.escape(response.topic)}</title>
      <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; line-height: 1.6; color: #111; }}
        h1, h2, h3 {{ color: #223344; }}
        p {{ margin: 0.8rem 0; }}
        ul {{ margin: 0.6rem 0 1rem 1.2rem; }}
        strong {{ font-weight: 700; }}
        .meta {{ padding: 1rem; background: #f4f7fb; border: 1px solid #dde3ea; margin-bottom: 1.2rem; }}
        .meta span {{ display: inline-block; margin-right: 1.5rem; }}
      </style>
    </head>
    <body>
      <div class=\"meta\">
        <span><strong>Topic:</strong> {html.escape(response.topic)}</span>
        <span><strong>Category:</strong> {html.escape(response.category)}</span>
        <span><strong>Source:</strong> {html.escape(response.source)}</span>
        <span><strong>Similarity:</strong> {response.similarity_score:.4f}</span>
      </div>
      {lecturette_html}
    </body>
    </html>
    """


def _build_html_form(topic: str = "") -> str:
    escaped_topic = html.escape(topic)
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"UTF-8\">
      <title>Generate Lecturette</title>
      <style>
        body {{ font-family: Arial, sans-serif; max-width: 760px; margin: 2rem auto; line-height: 1.6; color: #111; }}
        input[type=\"text\"] {{ width: 100%; padding: 0.8rem; margin: 0.6rem 0; border: 1px solid #bbb; border-radius: 4px; }}
        button {{ padding: 0.8rem 1.2rem; border: none; background: #2563eb; color: white; border-radius: 4px; cursor: pointer; }}
        button:hover {{ background: #1d4ed8; }}
      </style>
    </head>
    <body>
      <h1>Generate Lecturette</h1>
      <form method=\"get\">
        <label for=\"topic\">Enter topic:</label>
        <input id=\"topic\" name=\"topic\" value=\"{escaped_topic}\" placeholder=\"India-China relations\" required />
        <button type=\"submit\">Generate</button>
      </form>
      <p>Enter a topic and the page will render a student-ready lecturette.</p>
    </body>
    </html>
    """


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


@router.get("/generate-lecturette-html", response_class=HTMLResponse)
async def generate_lecturette_html(topic: str | None = None) -> HTMLResponse:
    if not topic:
        return HTMLResponse(_build_html_form())

    start = time.perf_counter()
    try:
        parsed = await ParserService.parse_topic(topic)
        main_topic = parsed["main_topic"]
        category = parsed["category"]
        keywords = parsed.get("keywords", [])
        normalized_query = parsed["normalized_query"]

        async with db_session_ctx() as db:
            retrieval = await RetrievalService.retrieve(
                db, main_topic, category, keywords, normalized_query
            )

            structured = await ResearchAgent.build_structure(
                topic=main_topic,
                category=category,
                research_summary=retrieval.research_entry.research_summary,
                compressed_research=retrieval.research_entry.compressed_research,
            )

            lecturette_md = await WriterAgent.write_lecturette(structured)

            lecturette_row = Lecturette(
                topic=main_topic,
                category=category,
                lecturette=lecturette_md,
                research_cache_id=retrieval.research_entry.id,
                similarity_score=retrieval.similarity_score,
                source="database" if retrieval.cache_hit else "web",
            )
            db.add(lecturette_row)

            elapsed_ms = (time.perf_counter() - start) * 1000
            db.add(
                SearchLog(
                    query=topic,
                    used_cache=retrieval.cache_hit,
                    web_search_performed=retrieval.web_search_performed,
                    similarity_score=retrieval.similarity_score,
                    latency_ms=elapsed_ms,
                )
            )

        response = LecturetteResponse(
            topic=main_topic,
            category=category,
            lecturette=lecturette_md,
            source="database" if retrieval.cache_hit else "web",
            similarity_score=round(retrieval.similarity_score, 4),
            saved=True,
            cache_hit=retrieval.cache_hit,
        )

        return HTMLResponse(_render_html_page(response))

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error generating lecturette for topic '%s'", topic)
        raise HTTPException(
            status_code=500, detail="Internal error generating lecturette."
        ) from exc
