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


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), store=Depends(get_store)) -> UploadResponse:
    """Upload a PDF, chunk it, embed it, and persist the updated store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / file.filename
    path.write_bytes(await file.read())
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
