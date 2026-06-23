"""
Pydantic schemas for the /generate-lecturette API contract.
"""

from pydantic import BaseModel, Field, field_validator


class LecturetteRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=300)

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Topic cannot be empty or whitespace.")
        return v


class LecturetteResponse(BaseModel):
    topic: str
    category: str
    lecturette: str
    source: str  # "database" | "web"
    similarity_score: float
    saved: bool
    cache_hit: bool


class HealthResponse(BaseModel):
    status: str
    database: str
    app_name: str
