"""Response and error schemas for POST /upload."""

from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Result of indexing one uploaded PDF.

    The first five fields are the original contract. The visual counts below
    them are additive: they report what the document contributed to the caption
    and CLIP streams, and stay 0 when no visual indices are configured.
    """

    doc_id: str
    source: str
    n_pages: int
    n_chunks: int
    already_indexed: bool = False

    # --- additive visual indexing counts (PR6) ---
    n_figures: int = 0
    n_caption_records: int = 0
    n_visual_records: int = 0
    visual_retrieval_available: bool = False


class UploadError(BaseModel):
    detail: str
    code: str  # "invalid_extension" | "file_too_large" | "invalid_mime" | "malformed_pdf"
