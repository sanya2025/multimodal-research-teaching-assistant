"""PDF ingestion — loads a PDF into typed PageRecord / PdfDocument objects."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from mrta.core.schemas import PageRecord, PdfDocument


def _doc_id(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}_{h}"


def load_pdf(path: str | Path) -> PdfDocument:
    path = Path(path)
    doc = fitz.open(path)
    doc_id = _doc_id(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text; "blocks" gives layout boxes
        n_images = len(page.get_images(full=True))
        pages.append(
            PageRecord(
                doc_id=doc_id,
                page=i,
                text=text,
                n_images=n_images,
                source=path.name,
            )
        )
    return PdfDocument(
        doc_id=doc_id,
        source=path.name,
        title=doc.metadata.get("title") or None,
        n_pages=doc.page_count,
        pages=pages,
    )


def ocr_page_if_needed(page: fitz.Page, dpi: int = 200) -> str:
    """Return page text, falling back to pytesseract OCR for scanned pages."""
    text = page.get_text().strip()
    if text:
        return text
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print("OCR fallback failed:", e)
        return ""
