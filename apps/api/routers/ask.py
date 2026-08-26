"""POST /ask — retrieve relevant chunks and generate a grounded answer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.deps import get_llm, get_retriever, get_store, get_vlm
from apps.api.schemas import AskRequest, AskResponse, SourceChunk, VisualSource
from mrta.core.rag_pipeline import rag_query
from mrta.generation.multimodal_rag import MultimodalRAG

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    store=Depends(get_store),
    llm=Depends(get_llm),
    retriever=Depends(get_retriever),
    vlm=Depends(get_vlm),
) -> AskResponse:
    """Ask a question; return a grounded answer with page citations.

    Set ``retrieval_mode="multimodal"`` to use the full text+visual RAG pipeline.
    Optionally pair with ``teaching_mode`` to shape the VLM's instructional style.
    """
    if req.retrieval_mode == "multimodal":
        if retriever is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Multimodal retriever not available. "
                    "Install mrta-rag[multimodal] and restart the server."
                ),
            )
        mmrag = MultimodalRAG(
            retriever=retriever,
            vlm=vlm,
            text_top_k=req.top_k,
            teaching_mode=req.teaching_mode,
        )
        result = mmrag.ask(req.question)
        text_sources = [
            SourceChunk(
                page=c.page,
                source=c.source,
                chunk_id=c.evidence_id,
                preview="",
                score=None,
            )
            for c in result.text_citations
        ]
        visual_sources = [
            VisualSource(
                label=c.label,
                page=c.page,
                source=c.source,
                figure_index=c.figure_index,
                modality=c.modality,
            )
            for c in result.visual_citations
        ]
        return AskResponse(
            answer=result.answer,
            sources=text_sources,
            latency_s=result.latency_s,
            retrieval_mode=result.retrieval_mode,
            visual_sources=visual_sources,
        )

    # text-only path (unchanged)
    result = rag_query(
        req.question, vector_store=store, llm=llm, top_k=req.top_k, source_filter=req.source
    )
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
