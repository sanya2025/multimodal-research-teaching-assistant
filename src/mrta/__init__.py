"""mrta — Multimodal AI Research & Teaching Assistant library.

Install options
---------------
Core only (config, schemas, LLM client, prompts):
    pip install mrta-rag

PDF ingestion:
    pip install "mrta-rag[pdf]"

Chunking, embeddings, and FAISS vector search:
    pip install "mrta-rag[retrieval]"

Full install:
    pip install "mrta-rag[all]"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core — always available with the base install
# ---------------------------------------------------------------------------
from mrta.core.config import Settings, settings
from mrta.core.exceptions import (
    EmbeddingError,
    EvaluationError,
    IngestionError,
    LLMError,
    MRTAError,
    RetrievalError,
)
from mrta.core.llm import LLMClient
from mrta.core.rag_pipeline import rag_query
from mrta.core.schemas import Chunk, EvalReport, FigureRecord, PageRecord, PdfDocument
from mrta.observability.logging import StructuredLogger
from mrta.prompts import MODES, load_prompt

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# PDF extra  —  pip install "mrta-rag[pdf]"
# ---------------------------------------------------------------------------
try:
    from mrta.ingestion.chunker import chunk_pdf
    from mrta.ingestion.figure_extractor import extract_figures
    from mrta.ingestion.pdf_loader import load_pdf
except ImportError:
    pass  # requires mrta-rag[pdf] (PyMuPDF, pdfplumber)

# ---------------------------------------------------------------------------
# Retrieval extra  —  pip install "mrta-rag[retrieval]"
# ---------------------------------------------------------------------------
try:
    from mrta.retrieval.embedder import Embedder
    from mrta.retrieval.vector_store import VectorStore
except ImportError:
    pass  # requires mrta-rag[retrieval] (sentence-transformers, faiss-cpu)

# ---------------------------------------------------------------------------
# Multimodal extra  —  pip install "mrta-rag[multimodal]"
# ---------------------------------------------------------------------------
try:
    from mrta.multimodal.clip_embedder import CLIPEmbedder
    from mrta.multimodal.vlm_client import VLMClient
except ImportError:
    pass  # requires mrta-rag[multimodal] (open-clip-torch)

# ---------------------------------------------------------------------------
# Eval extra  —  pip install "mrta-rag[eval]"
# ---------------------------------------------------------------------------
try:
    from mrta.evaluation.eval_pipeline import run_eval
    from mrta.evaluation.metrics import (
        answer_relevance,
        citation_correctness,
        faithfulness,
        hallucination_rate,
    )
except ImportError:
    pass  # requires mrta-rag[eval] (deepeval)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "__version__",
    # Exceptions
    "MRTAError",
    "IngestionError",
    "EmbeddingError",
    "RetrievalError",
    "LLMError",
    "EvaluationError",
    # Config
    "Settings",
    "settings",
    # Schemas
    "PageRecord",
    "PdfDocument",
    "Chunk",
    "FigureRecord",
    "EvalReport",
    # Core pipeline
    "LLMClient",
    "rag_query",
    # Prompts
    "load_prompt",
    "MODES",
    # Observability
    "StructuredLogger",
    # PDF extra
    "load_pdf",
    "chunk_pdf",
    "extract_figures",
    # Retrieval extra
    "Embedder",
    "VectorStore",
    # Multimodal extra
    "CLIPEmbedder",
    "VLMClient",
    # Eval extra
    "run_eval",
    "answer_relevance",
    "faithfulness",
    "citation_correctness",
    "hallucination_rate",
]
