"""mrta.ingestion.figure_extractor — extract embedded raster figures from a PDF."""

from __future__ import annotations

from pathlib import Path

from mrta.core.schemas import FigureRecord
from mrta.ingestion.pdf_loader import _doc_id

# Characters of surrounding page text kept as nearby_text context.
_NEARBY_TEXT_CHARS = 400


def extract_figures(pdf_path: str | Path) -> list[FigureRecord]:
    """Extract all embedded raster images from a PDF, one FigureRecord per image.

    CMYK pixmaps are converted to RGB before encoding. Vector-only figures are
    not captured (see docs/adr/ caveats for layout-model approach).

    Each record includes pixel dimensions, the image bounding box in PDF points,
    and a short excerpt of nearby page text for context.
    """
    import fitz  # noqa: PLC0415 — lazy: only needed when [pdf] extra is installed

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    did = _doc_id(pdf_path)
    figs: list[FigureRecord] = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        nearby = page_text[:_NEARBY_TEXT_CHARS].strip() if page_text else None

        img_list = page.get_images(full=True)
        for idx, img in enumerate(img_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:  # CMYK or other wide-gamut → convert to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # Bounding box of this image on the page (PDF point coordinates)
            bbox: tuple[float, float, float, float] | None = None
            for item in page.get_image_info(xrefs=True):
                if item.get("xref") == xref:
                    r = item["bbox"]
                    bbox = (r[0], r[1], r[2], r[3])
                    break

            figs.append(
                FigureRecord(
                    doc_id=did,
                    source=pdf_path.name,
                    page=page_num,
                    figure_index=idx,
                    image_bytes=pix.tobytes("png"),
                    width=pix.width,
                    height=pix.height,
                    bbox=bbox,
                    nearby_text=nearby,
                )
            )
            pix = None
    return figs
