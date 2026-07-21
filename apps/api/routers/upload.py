"""POST /upload — ingest a PDF and add its chunks to the vector store."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from apps.api.deps import get_store
from apps.api.schemas import UploadResponse
from mrta.core.config import settings
from mrta.ingestion.chunker import chunk_pdf
from mrta.ingestion.pdf_loader import load_pdf

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
PDF_MAGIC = b"%PDF"


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), store=Depends(get_store)) -> UploadResponse:
    """Upload a PDF, chunk it, embed it, and persist the updated store."""
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

    # 7. Parse — IngestionError mapped to 422 by the global exception handler in main.py
    pdf = load_pdf(path)
    chunks = chunk_pdf(pdf, strategy="recursive")
    store.add(chunks)
    store.save(settings.vector_store_path / "default")
    return UploadResponse(
        doc_id=pdf.doc_id,
        source=pdf.source,
        n_pages=pdf.n_pages,
        n_chunks=len(chunks),
    )
