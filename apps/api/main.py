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
from mrta.retrieval.embedder import Embedder
from mrta.retrieval.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = Embedder()
    store_dir = Path(settings.vector_store_path) / "default"
    if (store_dir / "index.faiss").exists():
        store = VectorStore.load(store_dir, embedder)
    else:
        store = VectorStore(embedder)
    app.state.store = store
    app.state.llm = LLMClient()
    app.state.embedder = embedder
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
