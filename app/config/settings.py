"""
Centralized application settings.
All values are loaded from environment variables (.env file in local dev,
real environment variables in production / Docker).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------------
    # App
    # ---------------------------------------------------------------
    APP_NAME: str = "SSB Lecturette Agent"
    ENV: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # ---------------------------------------------------------------
    # Database (PostgreSQL + pgvector)
    # ---------------------------------------------------------------
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/lecturette_db"
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False

    # ---------------------------------------------------------------
    # Gemini (used for both Research Agent & Writer Agent + embeddings)
    # ---------------------------------------------------------------
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    EMBEDDING_DIM: int = 768

    # ---------------------------------------------------------------
    # Google Search
    # ---------------------------------------------------------------
    GOOGLE_API_KEY: str = ""
    GOOGLE_CSE_ID: str = ""
    GOOGLE_MAX_RESULTS: int = 6

    # ---------------------------------------------------------------
    # Retrieval / Caching strategy
    # ---------------------------------------------------------------
    SIMILARITY_CACHE_HIT_THRESHOLD: float = 0.80
    SIMILARITY_DUPLICATE_THRESHOLD: float = 0.95
    TOP_K_SIMILAR_TOPICS: int = 5

    # ---------------------------------------------------------------
    # Generation limits
    # ---------------------------------------------------------------
    MAX_RESEARCH_TOKENS: int = 1800
    MAX_LECTURETTE_TOKENS: int = 2000
    REQUEST_TIMEOUT_SECONDS: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — read .env / env vars only once."""
    return Settings()


settings = get_settings()
