"""Figure endpoints — caption figures (POST /figures) and serve them (GET /figures/image)."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from apps.api.deps import get_store
from apps.api.schemas import FigureCaptionItem, FiguresRequest, FiguresResponse
from mrta.core.config import settings
from mrta.generation.canonical_rag import safe_image_path
from mrta.ingestion.document_indexer import FIGURES_SUBDIR
from mrta.ingestion.figure_extractor import extract_figures
from mrta.multimodal.vlm_client import VLMClient
from mrta.prompts import load_prompt

router = APIRouter()


@router.post("/figures", response_model=FiguresResponse)
def explain_figures(req: FiguresRequest) -> FiguresResponse:
    """Extract embedded raster figures from a PDF and caption each with the VLM.

    Pass ``pages`` to limit extraction to pages cited by a preceding /ask call.
    Returns an empty ``figures`` list (with ``vlm_available=False``) if the
    vision model is not installed — callers should show the pull command.
    """
    pdf_path = Path("data/raw") / req.source
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {req.source}. Upload it first via POST /upload.",
        )

    vlm_available = VLMClient.is_available()
    model_name = settings.ollama_vlm_model

    if not vlm_available:
        return FiguresResponse(
            source=req.source,
            figures=[],
            vlm_available=False,
            model=model_name,
        )

    figs = extract_figures(pdf_path)
    if req.pages is not None:
        page_set = set(req.pages)
        figs = [f for f in figs if f.page in page_set]

    if not figs:
        return FiguresResponse(
            source=req.source,
            figures=[],
            vlm_available=True,
            model=model_name,
        )

    vlm = VLMClient()
    prompt = load_prompt("explain")
    captions: list[FigureCaptionItem] = []
    t0 = time.perf_counter()
    for fig in figs:
        caption = vlm.caption(fig.to_pil(), prompt=prompt)
        captions.append(
            FigureCaptionItem(
                page=fig.page,
                figure_index=fig.figure_index,
                caption=caption,
            )
        )
    latency_s = time.perf_counter() - t0

    return FiguresResponse(
        source=req.source,
        figures=captions,
        vlm_available=True,
        model=model_name,
        latency_s=round(latency_s, 2),
    )


@router.get("/figures/image")
def figure_image(
    source: str = Query(..., description="PDF filename as returned by GET /documents"),
    page: int = Query(..., ge=1, description="1-indexed page number"),
    figure_index: int = Query(1, ge=1, description="1-indexed figure number within the page"),
    store=Depends(get_store),
) -> FileResponse:
    """Return the extracted PNG for one figure.

    Answers carry an ``image_path`` for each visual citation, but that path is a
    server-side filesystem location a browser cannot load, which is why figures
    could be cited but never shown. This endpoint closes that gap.

    The path is derived server-side from the deterministic asset naming used by
    ``mrta.ingestion.document_indexer.figure_asset_path`` and is never taken from
    the caller. A client-supplied path would make this a path-traversal surface;
    deriving it means the only client input is a filename resolved against the
    index plus two positive integers. ``safe_image_path`` then applies the same
    containment check the generation path uses.
    """
    doc_id = next((c.doc_id for c in store._chunks if c.source == source), None)
    if doc_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not indexed: {source}. Upload it first via POST /upload.",
        )

    relative = f"data/{FIGURES_SUBDIR}/{doc_id}_p{page}_f{figure_index}.png"
    safe = safe_image_path(relative)
    if safe is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No extracted figure for {source} page {page} figure {figure_index}. "
                "The page may contain only vector graphics, which the raster "
                "extractor does not capture."
            ),
        )

    return FileResponse(
        Path(safe),
        media_type="image/png",
        filename=f"{doc_id}_p{page}_f{figure_index}.png",
    )
