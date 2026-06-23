"""
Thin async wrapper around the Gemini API (google-genai SDK).
Centralizes model config, retries, and error handling so every
service/agent calls through one place — easy to swap providers later.
"""

import asyncio
import logging

from google import genai
from google.genai import types

from app.config.settings import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._text_model = settings.GEMINI_MODEL
        self._embedding_model = settings.GEMINI_EMBEDDING_MODEL

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.4,
        retries: int = 2,
    ) -> str:
        """Single-turn generation call. Returns raw text output."""
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._text_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    ),
                )
                return response.text or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 1.5 * (attempt + 1)
                logger.warning(
                    "Gemini generate_text attempt %d failed: %s (retrying in %.1fs)",
                    attempt + 1,
                    exc,
                    wait,
                )
                if attempt < retries:
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Gemini text generation failed after retries: {last_exc}")

    async def embed_text(self, text: str, retries: int = 2) -> list[float]:
        """Returns a single embedding vector for the given text."""
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = await asyncio.to_thread(
                    self._client.models.embed_content,
                    model=self._embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=settings.EMBEDDING_DIM
                    ),
                )
                return list(response.embeddings[0].values)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 1.5 * (attempt + 1)
                logger.warning(
                    "Gemini embed_text attempt %d failed: %s (retrying in %.1fs)",
                    attempt + 1,
                    exc,
                    wait,
                )
                if attempt < retries:
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Gemini embedding generation failed after retries: {last_exc}")


gemini_client = GeminiClient()
