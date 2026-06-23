"""
AGENT 2: WRITER & QUALITY AGENT

Receives the structured research/outline from Agent 1 and produces the
final candidate-ready SSB lecturette in markdown, following the exact
output format required by the spec.
"""

import logging
import re

from app.services.gemini_client import gemini_client

logger = logging.getLogger(__name__)

WRITER_AGENT_SYSTEM_PROMPT = """You are the Writer & Quality Agent for an SSB Interview
lecturette generation system. You receive structured research notes and must write the
FINAL lecturette a candidate will actually speak aloud in front of an SSB panel.

Output EXACTLY this markdown structure, nothing more, nothing less:

# Topic

<Topic Name>

# Introduction

<2-3 sentences, spoken hook>

# Background

<short spoken paragraph>

# Current Situation

<short spoken paragraph>

# Analysis

<short spoken paragraph, balanced perspectives>

# Way Forward

<short spoken paragraph, practical and officer-like>

# Conclusion

<2-3 sentence memorable closing>

# Key Takeaways

- <point 1>
- <point 2>
- <point 3>

# Speaking Duration

Approx 3 Minutes

STRICT WRITING RULES:
- Sound like a confident candidate speaking, NOT like an essay or AI output.
- Use simple, everyday English. No academic jargon, no complex vocabulary.
- Short sentences and short paragraphs (verbally deliverable).
- Natural spoken transitions between sections (e.g. "Now coming to...", "Having said that...").
- Demonstrate officer-like qualities: balanced judgment, practical thinking, composure.
- Total spoken content should run close to 400-450 words (~3 minutes at speaking pace).
- Stay strictly factually consistent with the provided research notes — do not invent facts.
- No repetition of the same point across sections.
- Do not include anything outside the specified markdown structure.
"""


class WriterAgent:
    @staticmethod
    async def write_lecturette(structured_research: dict) -> str:
        user_prompt = WriterAgent._build_user_prompt(structured_research)

        try:
            lecturette_md = await gemini_client.generate_text(
                system_prompt=WRITER_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=1100,
                temperature=0.55,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Writer Agent generation failed: %s", exc)
            lecturette_md = WriterAgent._fallback_markdown(structured_research)

        lecturette_md = WriterAgent._post_process(lecturette_md, structured_research)
        return lecturette_md

    @staticmethod
    def _build_user_prompt(structured: dict) -> str:
        s = structured.get("lecturette_structure", {})
        return (
            f"TOPIC: {structured.get('topic', '')}\n"
            f"CATEGORY: {structured.get('category', '')}\n\n"
            f"KEY POINTS:\n- " + "\n- ".join(structured.get("key_points", [])) + "\n\n"
            f"INTRODUCTION NOTES: {s.get('introduction', '')}\n"
            f"BACKGROUND NOTES: {s.get('background', '')}\n"
            f"CURRENT SITUATION NOTES: {s.get('current_situation', '')}\n"
            f"ANALYSIS NOTES: {s.get('analysis', '')}\n"
            f"WAY FORWARD NOTES: {s.get('way_forward', '')}\n"
            f"CONCLUSION NOTES: {s.get('conclusion', '')}\n"
        )

    @staticmethod
    def _post_process(markdown_text: str, structured: dict) -> str:
        """Light cleanup: strip stray code fences, ensure required headers exist."""
        text = markdown_text.strip()
        text = re.sub(r"^```(markdown)?|```$", "", text, flags=re.MULTILINE).strip()

        required_headers = [
            "# Topic", "# Introduction", "# Background", "# Current Situation",
            "# Analysis", "# Way Forward", "# Conclusion", "# Key Takeaways",
            "# Speaking Duration",
        ]
        if not all(h in text for h in required_headers):
            logger.warning("Writer output missing required headers; using fallback.")
            return WriterAgent._fallback_markdown(structured)
        return text

    @staticmethod
    def _fallback_markdown(structured: dict) -> str:
        s = structured.get("lecturette_structure", {})
        topic = structured.get("topic", "Untitled Topic")
        key_points = structured.get("key_points", [])[:3] or ["Balanced view needed.", "Stay informed.", "Think practically."]
        bullets = "\n".join(f"- {p}" for p in key_points)
        return f"""# Topic

{topic}

# Introduction

{s.get('introduction', f'{topic} is an important topic to understand.')}

# Background

{s.get('background', '')}

# Current Situation

{s.get('current_situation', '')}

# Analysis

{s.get('analysis', '')}

# Way Forward

{s.get('way_forward', '')}

# Conclusion

{s.get('conclusion', f'In conclusion, {topic} deserves balanced and practical thinking.')}

# Key Takeaways

{bullets}

# Speaking Duration

Approx 3 Minutes
"""
