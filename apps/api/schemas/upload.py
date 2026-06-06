"""Response schema for POST /upload."""

from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    source: str
    n_pages: int
    n_chunks: int
