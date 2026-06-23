"""
Topic Parser Service.

Turns a raw user-provided topic string into a normalized structure:
    {
        "main_topic": "...",
        "keywords": [...],
        "category": "...",
        "normalized_query": "..."
    }

Uses a single cheap Gemini call (gemini-2.5-flash-lite) with a strict
JSON-only system prompt. Falls back to a lightweight heuristic parser
if the model call fails, so the pipeline never hard-blocks on this step.
"""

import json
import logging
import re

from app.config.settings import settings
from app.services.gemini_client import gemini_client

logger = logging.getLogger(__name__)

SSB_CATEGORIES = [
    "International Relations",
    "Defence & Security",
    "Geopolitics",
    "Social Issues",
    "National Security",
    "Technology",
    "Leadership",
    "Psychology",
    "Abstract Topic",
    "Defence Current Affairs",
    "Economy",
    "Environment",
    "General Awareness",
]

PARSER_SYSTEM_PROMPT = f"""You are a topic-parsing engine for an SSB Interview lecturette system.
Given a raw topic string, return ONLY valid JSON (no markdown, no commentary) in this exact shape:

{{
  "main_topic": "<cleaned canonical topic name>",
  "keywords": ["<4 to 8 short keywords/entities relevant to the topic>"],
  "category": "<one of: {", ".join(SSB_CATEGORIES)}>",
  "normalized_query": "<lowercase, trimmed, deduplicated-whitespace version of main_topic>"
}}

Rules:
- keywords must be short (1-3 words each), no duplicates.
- category must be exactly one of the listed options, pick the closest match.
- Do not add any text outside the JSON object.
"""


def _heuristic_fallback(topic: str) -> dict:
    """Used only if the LLM call fails — keeps the pipeline alive."""
    cleaned = re.sub(r"\s+", " ", topic).strip()
    words = [w for w in re.split(r"[\s\-,]+", cleaned) if len(w) > 2]
    return {
        "main_topic": cleaned,
        "keywords": words[:6] if words else [cleaned],
        "category": "General Awareness",
        "normalized_query": cleaned.lower(),
    }


class ParserService:
    @staticmethod
    async def parse_topic(raw_topic: str) -> dict:
        raw_topic = raw_topic.strip()
        if not raw_topic:
            raise ValueError("Topic cannot be empty.")

        try:
            response_text = await gemini_client.generate_text(
                system_prompt=PARSER_SYSTEM_PROMPT,
                user_prompt=raw_topic,
                max_output_tokens=300,
                temperature=0.2,
            )
            parsed = _extract_json(response_text)
            parsed.setdefault("main_topic", raw_topic)
            parsed.setdefault("keywords", [])
            parsed.setdefault("category", "General Awareness")
            parsed.setdefault(
                "normalized_query", re.sub(r"\s+", " ", raw_topic).strip().lower()
            )
            return parsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parser LLM call failed (%s); using heuristic fallback.", exc)
            return _heuristic_fallback(raw_topic)


def _extract_json(text: str) -> dict:
    """Strip markdown fences if present and parse the first JSON object found."""
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in parser output: {text[:200]}")
    return json.loads(match.group(0))
