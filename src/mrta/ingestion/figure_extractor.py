"""mrta.ingestion.figure_extractor — extract embedded raster figures from a PDF."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from mrta.core.schemas import FigureRecord
from mrta.ingestion.pdf_loader import _doc_id


def extract_figures(pdf_path: str | Path) -> list[FigureRecord]:
    """Extract all embedded raster images from a PDF, one FigureRecord per image.

    CMYK pixmaps are converted to RGB before encoding. Vector-only figures are
    not captured (see production-ready.md caveats for layout-model approach).
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    did = _doc_id(pdf_path)
    figs: list[FigureRecord] = []
    for page_num, page in enumerate(doc, start=1):
        for idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:  # CMYK or other wide-gamut → convert to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            figs.append(
                FigureRecord(
                    doc_id=did,
                    source=pdf_path.name,
                    page=page_num,
                    figure_index=idx,
                    image_bytes=pix.tobytes("png"),
                )
            )
            pix = None
    return figs
