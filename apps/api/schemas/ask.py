"""Request/response schemas for POST /ask."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    source: str | None = Field(None, description="Restrict retrieval to this PDF filename")
    retrieval_mode: Literal["text", "multimodal"] = Field(
        "text",
        description="text=text-only RAG; multimodal=text+visual RAG via VLM",
    )
    teaching_mode: Literal["explain", "socratic", "quiz", "compare", "visual_evidence"] | None = (
        Field(None, description="Teaching mode; only used when retrieval_mode='multimodal'")
    )


class SourceChunk(BaseModel):
    page: int
    source: str  # PDF filename
    chunk_id: str
    preview: str  # first 200 chars of chunk text
    score: float | None = None  # cosine similarity in [0, 1]


class VisualSource(BaseModel):
    label: str  # "[V1]", "[V2]", …
    page: int
    source: str  # PDF filename
    figure_index: int | None = None
    modality: str  # "image" or "page"


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_s: float
    retrieval_mode: str = "text"
    visual_sources: list[VisualSource] = []
