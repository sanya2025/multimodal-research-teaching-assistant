"""mrta — Multimodal AI Research & Teaching Assistant library."""

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
from mrta.evaluation.eval_pipeline import run_eval
from mrta.evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    faithfulness,
    hallucination_rate,
)
from mrta.ingestion.chunker import chunk_pdf
from mrta.ingestion.figure_extractor import extract_figures
from mrta.ingestion.pdf_loader import load_pdf
from mrta.multimodal.clip_embedder import CLIPEmbedder
from mrta.multimodal.vlm_client import VLMClient
from mrta.observability.logging import StructuredLogger
from mrta.prompts import MODES, load_prompt
from mrta.retrieval.embedder import Embedder
from mrta.retrieval.vector_store import VectorStore

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MRTAError",
    "IngestionError",
    "EmbeddingError",
    "RetrievalError",
    "LLMError",
    "EvaluationError",
    "Settings",
    "settings",
    "PageRecord",
    "PdfDocument",
    "Chunk",
    "FigureRecord",
    "EvalReport",
    "load_pdf",
    "chunk_pdf",
    "extract_figures",
    "Embedder",
    "VectorStore",
    "LLMClient",
    "rag_query",
    "load_prompt",
    "MODES",
    "StructuredLogger",
    "CLIPEmbedder",
    "VLMClient",
    "run_eval",
    "answer_relevance",
    "faithfulness",
    "citation_correctness",
    "hallucination_rate",
]
