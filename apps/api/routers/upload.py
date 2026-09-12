"""POST /upload — ingest a PDF and add its chunks to the vector store."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from apps.api.deps import get_canonical_stack, get_store
from apps.api.schemas import UploadResponse
from mrta.core.config import settings
from mrta.ingestion.document_indexer import index_document

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
PDF_MAGIC = b"%PDF"


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    store=Depends(get_store),
    canonical=Depends(get_canonical_stack),
) -> UploadResponse:
    """Upload a PDF, index it into every configured retrieval stream, and persist.

    Text indexing is unconditional. When a canonical stack is configured, the
    document's figures are also captioned into the caption index and embedded
    into the CLIP index, so the uploaded document is immediately usable by the
    full Text + Caption + CLIP query path rather than degrading to text-only.
    """
    filename = file.filename or ""

    # 1. Extension check
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # 2. Read once — reused for size and magic-byte checks
    data = await file.read()

    # 3. Size limit
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

    # 4. PDF magic bytes
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status_code=415, detail="File does not appear to be a valid PDF.")

    # 5. Safe filename — strip any directory components
    safe_name = Path(filename).name

    # 6. Duplicate guard — if this source is already in the index, skip re-indexing
    indexed = {c.source for c in store._chunks}
    if safe_name in indexed:
        existing = [c for c in store._chunks if c.source == safe_name]
        n_pages = max(c.page for c in existing)
        return UploadResponse(
            doc_id=existing[0].doc_id,
            source=safe_name,
            n_pages=n_pages,
            n_chunks=len(existing),
            already_indexed=True,
        )

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / safe_name
    path.write_bytes(data)

    # 7. Index — IngestionError mapped to 422 by the global exception handler in main.py
    stack = canonical or {}
    result = index_document(
        path,
        text_store=store,
        caption_store=stack.get("caption_store"),
        image_store=stack.get("image_store"),
        analyzer=stack.get("analyzer"),
        store_root=Path(settings.vector_store_path),
    )
    return UploadResponse(
        doc_id=result.doc_id,
        source=result.source,
        n_pages=result.n_pages,
        n_chunks=result.n_chunks,
        n_figures=result.n_figures,
        n_caption_records=result.n_caption_records,
        n_visual_records=result.n_visual_records,
        visual_retrieval_available=result.visual_retrieval_available,
    )
