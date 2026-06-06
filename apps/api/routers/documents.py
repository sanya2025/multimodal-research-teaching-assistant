"""GET /documents — list all indexed documents with page and chunk counts."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.deps import get_store
from apps.api.schemas import DocumentInfo

router = APIRouter()


@router.get("/documents", response_model=list[DocumentInfo])
def documents(store=Depends(get_store)) -> list[DocumentInfo]:
    """Return one DocumentInfo per unique doc_id present in the vector store."""
    by_doc: dict[str, dict] = {}
    for chunk in store._chunks:
        entry = by_doc.setdefault(
            chunk.doc_id,
            {"doc_id": chunk.doc_id, "source": chunk.source, "pages": set(), "n_chunks": 0},
        )
        entry["pages"].add(chunk.page)
        entry["n_chunks"] += 1
    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            source=d["source"],
            n_pages=len(d["pages"]),
            n_chunks=d["n_chunks"],
        )
        for d in by_doc.values()
    ]
