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
        except Exception:
            app.state.retriever = None
            app.state.vlm = None
    else:
        app.state.retriever = None
        app.state.vlm = None

    yield


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
