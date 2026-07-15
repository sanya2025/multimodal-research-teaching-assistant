"""POST /figures — extract and caption figures from an indexed PDF."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from apps.api.schemas import FigureCaptionItem, FiguresRequest, FiguresResponse
from mrta.core.config import settings
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
    for fig in figs:
        caption = vlm.caption(fig.to_pil(), prompt=prompt)
        captions.append(
            FigureCaptionItem(
                page=fig.page,
                figure_index=fig.figure_index,
                caption=caption,
            )
        )

    return FiguresResponse(
        source=req.source,
        figures=captions,
        vlm_available=True,
        model=model_name,
    )
