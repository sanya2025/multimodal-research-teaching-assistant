"""Request/response schemas for POST /figures."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FiguresRequest(BaseModel):
    source: str = Field(..., description="PDF filename as returned by GET /documents")
    pages: list[int] | None = Field(
        None, description="Filter to these page numbers; None means all pages"
    )


class FigureCaptionItem(BaseModel):
    page: int
    figure_index: int  # 1-indexed per page
    caption: str


class FiguresResponse(BaseModel):
    source: str
    figures: list[FigureCaptionItem]
    vlm_available: bool
    model: str  # ollama_vlm_model name, so callers can show the pull command
