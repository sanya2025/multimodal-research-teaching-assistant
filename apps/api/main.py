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
#
# The CLIP encoder must be mrta.retrieval.clip_embedder, not the mrta.multimodal
# one: ImageStore embeds figures by path and relies on warmup(). Passing the
# other encoder silently produced an empty CLIP index for months (ADR-014 §1).
try:
    from mrta.multimodal.vlm_client import VLMClient as _VLMClient
    from mrta.retrieval.clip_embedder import CLIPEmbedder as _CLIPEmbedder
    from mrta.retrieval.image_store_adapter import ImageStoreAdapter as _ImageStoreAdapter
    from mrta.retrieval.multimodal_retriever import MultimodalRetriever as _MultimodalRetriever

    _MULTIMODAL_AVAILABLE = True
except ImportError:
    _MULTIMODAL_AVAILABLE = False


def _warm_torch_runtime():
    """Load the CLIP encoder and run one forward pass, before FAISS is touched.

    Returns the warmed encoder, or None when the multimodal extra is absent or
    the weights cannot be loaded (offline, not yet downloaded). None disables the
    multimodal stack; text retrieval is unaffected.

    Ordering is load-bearing, not stylistic. faiss-cpu and torch both link
    against libomp, and whichever initializes the OpenMP runtime first wins. If
    FAISS gets there first, torch's next forward pass segfaults on macOS — the
    process dies with SIGSEGV, so no exception handler here or in the
    per-stream degradation logic can catch it. In this repository's Python 3.14
    environment the crash reproduced 5/5 with the old ordering and 0/5 with
    this one.

    ``CLIPEmbedder.warmup()`` exists for exactly this and documents it, but it
    was only ever called from ``ImageStore._ensure_index()`` — which runs long
    after the API lifespan has already loaded the text index through FAISS. By
    then the protection is lost. Calling it here, before any FAISS use, is what
    makes it effective. See ADR-014 §8.

    ``OMP_NUM_THREADS=1`` also avoids the crash, but serializes all OpenMP work
    for embedding and reranking, so it is documented as a fallback rather than
    used as the fix.
    """
    if not _MULTIMODAL_AVAILABLE:
        return None
    try:
        clip = _CLIPEmbedder()
        clip.warmup()
        return clip
    except Exception:
        # No CLIP means no visual streams. The API still serves text RAG, which
        # is the only path that has a required index.
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_tracing:
        configure_tracer(
            service_name=settings.otel_service_name,
            console=settings.otel_console_exporter,
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        )

    # Must happen before anything touches FAISS. See _warm_torch_runtime.
    clip = _warm_torch_runtime()

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
    if _MULTIMODAL_AVAILABLE and clip is not None:
        try:
            app.state.vlm = _VLMClient()

            canonical_stack = (
                _build_canonical_stack(embedder, clip)
                if settings.enable_canonical_retrieval
                else None
            )
            app.state.canonical_stack = canonical_stack

            # The legacy retriever reads the same persisted indices the canonical
            # stack loads. It previously received an empty, never-loaded
            # VisualVectorStore, so teaching modes — which route to this
            # retriever — could not return visual evidence at all.
            app.state.retriever = _build_legacy_retriever(store, canonical_stack)
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


def _source_for_doc(store: VectorStore, doc_id: str) -> str | None:
    """Resolve a document_id to its PDF filename using the loaded text index.

    The CLIP index stores ``VisualRecord``, which carries canonical identity but
    no filename, while citations are displayed by filename. The text index is the
    one stream that is always present, which makes it the natural lookup. Resolved
    per call rather than snapshotted at startup, so a document uploaded into a
    running server still cites its filename.
    """
    for chunk in store._chunks:
        if chunk.doc_id == doc_id:
            return chunk.source
    return None


def _build_legacy_retriever(store: VectorStore, canonical_stack: dict | None):
    """Wire MultimodalRetriever onto the persisted caption and CLIP indices.

    Teaching modes route to this retriever rather than the canonical pipeline
    (see ``apps.api.routers.ask``), because their prompt templates consume
    ``EvidenceRecord`` lists. That routing is unchanged; what changes is that the
    retriever is now given the visual streams that actually hold data.

    The CLIP stream is attached only when the index is non-empty: an empty
    adapter would add a stream that contributes nothing to RRF while still
    costing a query embedding on every request.
    """
    stack = canonical_stack or {}
    image_store = stack.get("image_store")

    visual_store = None
    if image_store is not None and image_store.size > 0:
        visual_store = _ImageStoreAdapter(
            image_store,
            source_resolver=lambda doc_id: _source_for_doc(store, doc_id),
        )

    return _MultimodalRetriever(
        vector_store=store,
        caption_store=stack.get("caption_store"),
        visual_store=visual_store,
    )


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
