"""Request/response schemas for POST /ask."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    source: str | None = Field(None, description="Restrict retrieval to this PDF filename")


class SourceChunk(BaseModel):
    page: int
    source: str  # PDF filename, e.g. "attention_is_all_you_need.pdf"
    chunk_id: str
    preview: str  # first 200 chars of Chunk.text
    score: float | None = None  # cosine similarity in [0, 1]


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_s: float
