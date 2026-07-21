"""POST /ask — retrieve relevant chunks and generate a grounded answer."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.deps import get_llm, get_store
from apps.api.schemas import AskRequest, AskResponse, SourceChunk
from mrta.core.rag_pipeline import rag_query

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, store=Depends(get_store), llm=Depends(get_llm)) -> AskResponse:
    """Ask a question; return a grounded answer with page citations."""
    result = rag_query(req.question, vector_store=store, llm=llm, top_k=req.top_k, source_filter=req.source)
    scores = result.get("scores", [])
    sources = [
        SourceChunk(
            page=c.page,
            source=c.source,
            chunk_id=c.chunk_id,
            preview=c.text[:200],
            score=scores[i] if i < len(scores) else None,
        )
        for i, c in enumerate(result["sources"])
    ]
    return AskResponse(answer=result["answer"], sources=sources, latency_s=result["latency_s"])
