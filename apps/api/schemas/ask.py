"""Request/response schemas for POST /ask."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class SourceChunk(BaseModel):
    page: int
    chunk_id: str
    preview: str  # first 200 chars of Chunk.text


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_s: float
