"""mrta.ingestion.page_renderer — render PDF pages as raster images.

Complements figure_extractor.py by capturing vector graphics and full-page
layout that raster image extraction misses. Returns EvidenceRecord instances
with modality="page" so they can feed into the visual retrieval pipeline.
"""

from __future__ import annotations

from pathlib import Path

from mrta.core.schemas import EvidenceRecord
from mrta.ingestion.pdf_loader import _doc_id


def render_page(
    pdf_path: str | Path,
    page_number: int,
    dpi: int = 150,
) -> EvidenceRecord:
    """Render a single PDF page to a PNG-encoded EvidenceRecord.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-indexed page number.
        dpi: Rendering resolution. 150 is a reasonable default; 200+ for detail.

    Returns:
        EvidenceRecord with modality="page" and image_bytes set to PNG data.
    """
    import fitz  # noqa: PLC0415 — lazy: only when [pdf] extra is installed

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
        raise ValueError(
            f"page_number {page_number} out of range for {pdf_path.name} " f"({len(doc)} pages)"
        )

    page = doc[page_number - 1]
    scale = dpi / 72.0  # PyMuPDF default resolution is 72 dpi
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")

    did = _doc_id(pdf_path)
    eid = f"{did}_p{page_number}_page"
    return EvidenceRecord(
        evidence_id=eid,
        doc_id=did,
        source=pdf_path.name,
        page=page_number,
        modality="page",
        image_bytes=png_bytes,
    )


def render_pages(
    pdf_path: str | Path,
    dpi: int = 150,
    pages: list[int] | None = None,
) -> list[EvidenceRecord]:
    """Render multiple PDF pages, returning one EvidenceRecord per page.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering resolution.
        pages: 1-indexed page numbers to render. None renders all pages.

    Returns:
        List of EvidenceRecord instances ordered by page number.
    """
    import fitz  # noqa: PLC0415

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    targets = pages if pages is not None else list(range(1, n_pages + 1))

    results: list[EvidenceRecord] = []
    for pn in targets:
        results.append(render_page(pdf_path, pn, dpi=dpi))
    return results
