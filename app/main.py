"""
Application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Run via Docker:
    docker compose up --build
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.lecturette import router as lecturette_router
from app.config.settings import settings
from app.database.connection import close_db, engine, init_db
from app.schemas.lecturette_schema import HealthResponse

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.ENV)
    await init_db()
    yield
    await close_db()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Cost-optimized, retrieval-first SSB lecturette generation system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lecturette_router, prefix=settings.API_PREFIX)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    db_status = "ok"
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Health check DB failure: %s", exc)
        db_status = "unreachable"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        database=db_status,
        app_name=settings.APP_NAME,
    )


@app.get("/")
async def root() -> dict:
    return {
        "app": settings.APP_NAME,
        "docs": "/docs",
        "generate_endpoint": f"{settings.API_PREFIX}/generate-lecturette",
    }
