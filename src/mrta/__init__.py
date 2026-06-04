"""mrta — Multimodal AI Research & Teaching Assistant library."""

from mrta.core.config import Settings, settings
from mrta.core.schemas import PageRecord, PdfDocument
from mrta.ingestion.pdf_loader import load_pdf

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Settings",
    "settings",
    "PageRecord",
    "PdfDocument",
    "load_pdf",
]
