"""Shared Pydantic models used across mrta modules."""

from __future__ import annotations

from pydantic import BaseModel


class PageRecord(BaseModel):
    doc_id: str
    page: int
    text: str
    n_images: int
    source: str


class PdfDocument(BaseModel):
    doc_id: str
    source: str
    title: str | None
    n_pages: int
    pages: list[PageRecord]


class Chunk(BaseModel):
    chunk_id: str  # "{doc_id}_p{page}_c{idx}"
    doc_id: str
    source: str
    page: int
    text: str
    section: str | None = None
    n_tokens: int | None = None
