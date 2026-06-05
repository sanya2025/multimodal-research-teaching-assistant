"""mrta — Multimodal AI Research & Teaching Assistant library."""

from mrta.core.config import Settings, settings
from mrta.core.llm import LLMClient
from mrta.core.rag_pipeline import rag_query
from mrta.core.schemas import Chunk, PageRecord, PdfDocument
from mrta.ingestion.chunker import chunk_pdf
from mrta.ingestion.pdf_loader import load_pdf
from mrta.observability.logging import StructuredLogger
from mrta.prompts import load_prompt
from mrta.retrieval.embedder import Embedder
from mrta.retrieval.vector_store import VectorStore

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Settings",
    "settings",
    "PageRecord",
    "PdfDocument",
    "Chunk",
    "load_pdf",
    "chunk_pdf",
    "Embedder",
    "VectorStore",
    "LLMClient",
    "rag_query",
    "load_prompt",
    "StructuredLogger",
]
