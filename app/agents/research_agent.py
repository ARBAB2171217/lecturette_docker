"""
AGENT 1: RESEARCH & STRUCTURE AGENT

Takes the compressed research (already retrieved by RetrievalService —
either from cache or fresh Google Search) and turns it into the
structured research notes + lecturette outline that Agent 2 consumes.

This agent does NOT decide whether to hit the web — that decision lives
in RetrievalService. This agent's job is purely structuring/compression
of whatever research it's handed, using one cheap Gemini call.
"""

import json
import logging
import re

from app.services.gemini_client import gemini_client

logger = logging.getLogger(__name__)

RESEARCH_AGENT_SYSTEM_PROMPT = """You are the Research & Structure Agent for an SSB Interview
lecturette generation system. You receive raw/compressed research material about a topic
and must produce ONLY valid JSON (no markdown, no commentary) in this exact shape:

{
  "topic": "<topic name>",
  "category": "<category>",
  "research_summary": "<150-250 word neutral, factual summary>",
  "key_points": ["<5 to 8 concise bullet-style key points>"],
  "lecturette_structure": {
      "introduction": "<1-2 sentence hook framing why this topic matters>",
      "background": "<short context/history>",
      "current_situation": "<what's happening now, current developments>",
      "analysis": "<balanced analysis - multiple angles, pros/cons, stakeholder views>",
      "way_forward": "<practical, officer-like recommendations>",
      "conclusion": "<short, memorable closing line>"
  }
}

Rules:
- Be factually grounded in the provided research only; do not invent statistics.
- Keep language simple, suitable for verbal delivery in ~3 minutes (~600-650 words total
  once written out by the Writer Agent).
- Show balanced, officer-like thinking (consider multiple perspectives).
- Do not include any text outside the JSON object.
"""


class ResearchAgent:
    @staticmethod
    async def build_structure(
        topic: str,
        category: str,
        research_summary: str,
        compressed_research: dict,
    ) -> dict:
        user_prompt = (
            f"TOPIC: {topic}\n"
            f"CATEGORY: {category}\n\n"
            f"RESEARCH SUMMARY:\n{research_summary}\n\n"
            f"COMPRESSED RESEARCH (JSON):\n{json.dumps(compressed_research, ensure_ascii=False)}\n"
        )

        try:
            raw = await gemini_client.generate_text(
                system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=900,
                temperature=0.3,
            )
            structured = _extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("Research Agent structuring failed: %s", exc)
            structured = _fallback_structure(topic, category, research_summary)

        structured.setdefault("topic", topic)
        structured.setdefault("category", category)
        structured.setdefault("research_summary", research_summary)
        structured.setdefault("key_points", [])
        structured.setdefault(
            "lecturette_structure",
            _fallback_structure(topic, category, research_summary)["lecturette_structure"],
        )
        return structured


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in research agent output: {text[:200]}")
    return json.loads(match.group(0))


def _fallback_structure(topic: str, category: str, research_summary: str) -> dict:
    """Minimal safe fallback so the pipeline never crashes if the LLM call fails."""
    return {
        "topic": topic,
        "category": category,
        "research_summary": research_summary,
        "key_points": [research_summary[:150]] if research_summary else [],
        "lecturette_structure": {
            "introduction": f"{topic} is a subject of significant relevance today.",
            "background": research_summary[:300] if research_summary else "",
            "current_situation": research_summary[300:600] if research_summary else "",
            "analysis": "This issue has multiple dimensions worth considering carefully.",
            "way_forward": "A balanced, pragmatic approach is needed going forward.",
            "conclusion": f"In conclusion, {topic} remains an important area for officers to understand.",
        },
    }
