"""POST /ask — retrieve relevant chunks and generate a grounded answer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.deps import (
    get_canonical_stack,
    get_llm,
    get_retriever,
    get_store,
    get_vlm,
)
from apps.api.schemas import AskRequest, AskResponse, SourceChunk, VisualSource
from mrta.core.rag_pipeline import rag_query
from mrta.core.schemas import MultimodalAnswer
from mrta.generation.multimodal_rag import MultimodalRAG

router = APIRouter()


def _to_response(result: MultimodalAnswer) -> AskResponse:
    """Map a multimodal answer onto the /ask response contract.

    Both multimodal paths funnel through here so the response shape cannot
    drift between them. Citations are built from retrieved evidence, so every
    field emitted here traces back to something actually retrieved.
    """
    text_sources = [
        SourceChunk(
            page=c.page,
            source=c.source,
            chunk_id=c.chunk_id or c.evidence_id,
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
            document_id=c.document_id,
            figure_id=c.figure_id,
            image_path=c.image_path,
            caption=c.caption,
            modality_sources=c.modality_sources,
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


@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    store=Depends(get_store),
    llm=Depends(get_llm),
    retriever=Depends(get_retriever),
    vlm=Depends(get_vlm),
    canonical=Depends(get_canonical_stack),
) -> AskResponse:
    """Ask a question; return a grounded answer with page citations.

    Set ``retrieval_mode="multimodal"`` to use the full text+visual RAG pipeline.
    Optionally pair with ``teaching_mode`` to shape the VLM's instructional style.

    When the canonical stack (PR4 fusion + PR5 reranking) is configured, the
    multimodal mode runs it. Otherwise it falls back to the legacy fused
    retriever. The response contract is identical either way.
    """
    if req.retrieval_mode == "multimodal":
        # Availability gate first, unchanged from before PR6: a configured
        # multimodal stack is what distinguishes 200 from 503.
        if retriever is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Multimodal retriever not available. "
                    "Install mrta-rag[multimodal] and restart the server."
                ),
            )

        # Given availability, prefer the evaluated canonical stack. Teaching
        # modes stay on the legacy path: their templates consume EvidenceRecord
        # lists, so routing them here would change their rendered prompts.
        if canonical is not None and req.teaching_mode is None:
            from mrta.generation.canonical_rag import CanonicalMultimodalRAG

            canonical_rag = CanonicalMultimodalRAG(
                text_store=store,
                vlm=vlm,
                caption_store=canonical.get("caption_store"),
                image_store=canonical.get("image_store"),
                reranker=canonical.get("reranker"),
                top_k=req.top_k,
            )
            return _to_response(canonical_rag.ask(req.question))

        mmrag = MultimodalRAG(
            retriever=retriever,
            vlm=vlm,
            text_top_k=req.top_k,
            teaching_mode=req.teaching_mode,
        )
        return _to_response(mmrag.ask(req.question))

    # text-only path (unchanged)
    text_result: dict = rag_query(
        req.question, vector_store=store, llm=llm, top_k=req.top_k, source_filter=req.source
    )
    scores = text_result.get("scores", [])
    sources = [
        SourceChunk(
            page=c.page,
            source=c.source,
            chunk_id=c.chunk_id,
            preview=c.text[:200],
            score=scores[i] if i < len(scores) else None,
        )
        for i, c in enumerate(text_result["sources"])
    ]
    return AskResponse(
        answer=text_result["answer"], sources=sources, latency_s=text_result["latency_s"]
    )
