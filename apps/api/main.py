"""FastAPI entry point. Run with: uvicorn apps.api.main:app --reload --port 8000"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from apps.api.routers import ask as ask_router
from apps.api.routers import documents as documents_router
from apps.api.routers import figures as figures_router
from apps.api.routers import upload as upload_router
from mrta.core.config import settings
from mrta.core.exceptions import IngestionError
from mrta.core.llm import LLMClient
from mrta.observability.tracing import configure_tracer
from mrta.retrieval.embedder import Embedder
from mrta.retrieval.vector_store import VectorStore

# optional multimodal stack — requires mrta-rag[multimodal]
try:
    from mrta.multimodal.clip_embedder import CLIPEmbedder as _CLIPEmbedder
    from mrta.multimodal.vlm_client import VLMClient as _VLMClient
    from mrta.retrieval.multimodal_retriever import MultimodalRetriever as _MultimodalRetriever
    from mrta.retrieval.visual_vector_store import VisualVectorStore as _VisualVectorStore

    _MULTIMODAL_AVAILABLE = True
except ImportError:
    _MULTIMODAL_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_tracing:
        configure_tracer(
            service_name=settings.otel_service_name,
            console=settings.otel_console_exporter,
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        )

    embedder = Embedder()

    store_dir = Path(settings.vector_store_path) / "default"

    if (store_dir / "index.faiss").exists():
        store = VectorStore.load(store_dir, embedder)
    else:
        store = VectorStore(embedder)

    app.state.store = store
    app.state.llm = LLMClient()
    app.state.embedder = embedder

    # multimodal stack (optional)
    if _MULTIMODAL_AVAILABLE:
        try:
            clip = _CLIPEmbedder()
            visual_store = _VisualVectorStore(clip)
            app.state.retriever = _MultimodalRetriever(
                vector_store=store, visual_store=visual_store
            )
            app.state.vlm = _VLMClient()
            app.state.canonical_stack = (
                _build_canonical_stack(embedder, clip)
                if settings.enable_canonical_retrieval
                else None
            )
        except Exception:
            app.state.retriever = None
            app.state.vlm = None
            app.state.canonical_stack = None
    else:
        app.state.retriever = None
        app.state.vlm = None
        app.state.canonical_stack = None

    yield


def _build_canonical_stack(embedder, clip) -> dict | None:
    """Assemble the PR4/PR5 canonical components, tolerating missing pieces.

    Each component is loaded independently, because the canonical pipeline
    degrades per-stream: a missing caption index costs the caption stream, not
    the query.

    The caption and CLIP indices are loaded from the same persisted location
    that production ingestion writes (``mrta.ingestion.document_indexer``), so
    a document uploaded in an earlier process is retrievable after a restart.
    An index that does not exist yet starts empty and is populated by the next
    upload.

    The analyzer is included so ``/upload`` can caption figures with the same
    configured VLM the rest of the stack uses.
    """
    from mrta.ingestion.document_indexer import CAPTION_INDEX_NAME, CLIP_INDEX_NAME
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.image_store import ImageStore

    store_root = Path(settings.vector_store_path)
    stack: dict = {
        "caption_store": None,
        "image_store": None,
        "reranker": None,
        "analyzer": None,
    }

    caption_dir = store_root / CAPTION_INDEX_NAME
    try:
        stack["caption_store"] = (
            CaptionVectorStore.load(caption_dir, embedder)
            if (caption_dir / "index.faiss").exists()
            else CaptionVectorStore(embedder)
        )
    except Exception:
        stack["caption_store"] = None

    clip_dir = store_root / CLIP_INDEX_NAME
    try:
        stack["image_store"] = (
            ImageStore.load(clip_dir, clip)
            if (clip_dir / "index.faiss").exists()
            else ImageStore(clip)
        )
    except Exception:
        stack["image_store"] = None

    try:
        from mrta.multimodal.visual_analyzer import VisualAnalyzer

        stack["analyzer"] = VisualAnalyzer()
    except Exception:
        # Captioning unavailable: figures are still indexed and fall back to
        # their nearby page text. No caption is fabricated.
        stack["analyzer"] = None

    if settings.enable_cross_encoder_rerank:
        try:
            from mrta.retrieval.reranker import CrossEncoderReranker

            stack["reranker"] = CrossEncoderReranker()
        except Exception:
            # Cross-encoder weights unavailable (offline, or model not
            # downloaded). The pipeline falls back to canonical RRF ordering.
            stack["reranker"] = None

    return stack


app = FastAPI(
    title="Multimodal AI Research & Teaching Assistant",
    version="0.1.0",
    description="Upload PDFs, ask grounded questions, explain figures.",
    lifespan=lifespan,
)


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "code": "malformed_pdf"},
    )


app.include_router(ask_router.router)
app.include_router(upload_router.router)
app.include_router(documents_router.router)
app.include_router(figures_router.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
