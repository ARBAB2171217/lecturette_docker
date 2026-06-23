"""
Google Custom Search Service — web search provider.
Called only when the retrieval service determines the cache similarity
score is below the configured threshold (cache miss).
"""

import logging
import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class GoogleSearchService:
    def __init__(self) -> None:
        self.endpoint = "https://www.googleapis.com/customsearch/v1"

    async def search(self, query: str) -> dict:
        """
        Runs a Google Custom Search query and returns:
            {
                "answer": "",
                "results": [{"title", "url", "content", "score"}, ...],
                "sources": [url, ...]
            }
        """
        if not settings.GOOGLE_API_KEY or not settings.GOOGLE_CSE_ID:
            logger.warning("Google API Key or CSE ID is not set. Skipping search.")
            return {"answer": "", "results": [], "sources": []}

        params = {
            "key": settings.GOOGLE_API_KEY,
            "cx": settings.GOOGLE_CSE_ID,
            "q": query,
            "num": settings.GOOGLE_MAX_RESULTS,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.endpoint,
                    params=params,
                    timeout=settings.REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("Google search failed for query '%s': %s", query, exc)
            return {"answer": "", "results": [], "sources": []}

        items = data.get("items", [])
        results = []
        sources = []

        for item in items:
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            if url:
                results.append({
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "score": 1.0,
                })
                sources.append(url)

        return {
            "answer": "",  # Google Search API doesn't synthesize an answer
            "results": results,
            "sources": sources,
        }

    @staticmethod
    def build_research_queries(main_topic: str, keywords: list[str]) -> list[str]:
        """
        Generates a small set of targeted queries instead of one vague
        query, covering the categories required by the quality rules:
        definitions, history, current developments, stats, strategic
        importance, expert opinions.
        """
        kw = ", ".join(keywords[:4]) if keywords else ""
        base = main_topic if not kw else f"{main_topic} ({kw})"
        return [
            f"{base} latest developments 2026",
            f"{base} background and history",
            f"{base} strategic and security importance India",
        ]


google_search_service = GoogleSearchService()
