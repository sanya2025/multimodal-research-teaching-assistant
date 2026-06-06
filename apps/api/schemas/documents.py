"""Response schema for GET /documents."""

from __future__ import annotations

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    n_pages: int
    n_chunks: int
