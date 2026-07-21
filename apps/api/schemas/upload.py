"""Response and error schemas for POST /upload."""

from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    source: str
    n_pages: int
    n_chunks: int
    already_indexed: bool = False


class UploadError(BaseModel):
    detail: str
    code: str  # "invalid_extension" | "file_too_large" | "invalid_mime" | "malformed_pdf"
