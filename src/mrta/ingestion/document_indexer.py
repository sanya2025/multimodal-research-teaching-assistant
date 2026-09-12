"""mrta.ingestion.document_indexer — build every production retrieval artifact.

One uploaded PDF must produce everything the canonical query path consumes::

    PDF ─┬─► text chunks     ──► VectorStore        (text stream)
         └─► figure records  ─┬─► PNG asset on disk
                              ├─► VisualAnalyzer ──► CaptionVectorStore (caption stream)
                              └─► VisualRecord   ──► ImageStore         (CLIP stream)

Before this module, production ingestion built only the text index, so a newly
uploaded document silently degraded to text-only retrieval while the caption and
CLIP indices existed solely as benchmark artifacts.

This is integration, not new retrieval: chunking, figure extraction, captioning,
embedding and indexing are all existing components, invoked in order. No model,
parameter or retrieval behaviour is chosen here.

Canonical identity
------------------
Figure identity comes from `(doc_id, page, figure_index)`, which
``extract_figures`` already assigns and ``FigureRecord.to_evidence_record``
already encodes. Both the caption record and the CLIP record for one physical
figure are built from the same ``FigureRecord``, so ``canonical_identity()``
merges them without consulting any benchmark manifest.

Failure isolation
-----------------
A document must not be lost because one figure is unreadable. The text index is
written first and independently; each figure is then processed in its own
try/except, and a failure is recorded in ``figure_failures`` rather than raised.
Captioning already degrades internally (``VisualAnalyzer`` returns an empty
description rather than raising when the VLM is unavailable), and this module
preserves the established nearby-text fallback behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mrta.core.schemas import Chunk, EvidenceRecord, FigureRecord, VisualRecord
from mrta.observability.tracing import trace_span

if TYPE_CHECKING:
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.image_store import ImageStore
    from mrta.retrieval.vector_store import VectorStore

# Where production keeps its artifacts, all relative to the working directory so
# an emitted image_path stays inside the `data/` tree the API is willing to
# serve (see mrta.generation.canonical_rag.safe_image_path).
FIGURES_SUBDIR = "figures"
TEXT_INDEX_NAME = "default"
CAPTION_INDEX_NAME = "captions"
CLIP_INDEX_NAME = "clip_images"

# caption_source provenance values, matching the evaluation builder's vocabulary
# so production and benchmark provenance can be compared directly.
CAPTION_SOURCE_VLM = "vlm_generated"
CAPTION_SOURCE_FALLBACK = "nearby_text_fallback"
CAPTION_SOURCE_EMPTY = "empty"

EXTRACTION_RASTER_CROP = "raster_crop"


def production_figure_id(page: int, figure_index: int | None) -> str:
    """Canonical figure id derived from production ingestion metadata.

    Re-exported from the retrieval layer so ingestion and retrieval cannot drift
    apart: if this derivation changed on one side only, caption and CLIP records
    for the same figure would stop merging.
    """
    from mrta.retrieval.canonical_pipeline import production_figure_id as _derive

    return _derive(page, figure_index)


@dataclass
class IngestionResult:
    """What one ingested document actually produced."""

    doc_id: str
    source: str
    n_pages: int
    n_chunks: int
    n_figures: int = 0
    n_caption_records: int = 0
    n_visual_records: int = 0
    caption_sources: dict[str, int] = field(default_factory=dict)
    figure_failures: list[str] = field(default_factory=list)
    degraded: dict[str, str] = field(default_factory=dict)

    @property
    def visual_retrieval_available(self) -> bool:
        """True when this document contributed to at least one visual stream."""
        return self.n_caption_records > 0 or self.n_visual_records > 0


def figure_asset_path(record: FigureRecord, figures_dir: Path) -> Path:
    """Deterministic on-disk location for one figure's PNG."""
    return figures_dir / f"{record.doc_id}_p{record.page}_f{record.figure_index}.png"


def build_caption_evidence(
    record: FigureRecord,
    image_path: str,
    analyzer: Any | None,
) -> tuple[EvidenceRecord, str]:
    """Build the caption-index record for one figure, with provenance.

    Returns the EvidenceRecord and its ``caption_source``. Captions are never
    fabricated: when the analyzer produces nothing usable, the record falls back
    to the nearby page text captured at extraction time, and says so.
    """
    evidence = record.to_evidence_record()
    evidence.image_path = image_path

    if analyzer is not None:
        # VisualAnalyzer swallows VLM failures internally and returns an empty
        # description, so this populates caption fields or leaves them None.
        analyzer.analyze_evidence(evidence)

    if evidence.retrieval_text():
        caption_source = (
            CAPTION_SOURCE_VLM
            if (evidence.caption or evidence.detailed_description)
            else CAPTION_SOURCE_FALLBACK
        )
    elif evidence.nearby_text:
        caption_source = CAPTION_SOURCE_FALLBACK
    else:
        caption_source = CAPTION_SOURCE_EMPTY

    # Persisted on the record, not just tallied on the result, so the caption's
    # origin is still recoverable from the index after a restart.
    evidence.caption_source = caption_source
    evidence.extraction_method = EXTRACTION_RASTER_CROP

    return evidence, caption_source


def build_visual_record(record: FigureRecord, image_path: str) -> VisualRecord:
    """Build the CLIP-index record for one figure.

    ``figure_id`` is the derived production id, so this record and the caption
    record above resolve to the same canonical identity.
    """
    return VisualRecord(
        document_id=record.doc_id,
        page=record.page,
        figure_id=production_figure_id(record.page, record.figure_index),
        figure_index=record.figure_index,
        image_path=image_path,
        extraction_method=EXTRACTION_RASTER_CROP,
    )


def index_document(
    pdf_path: str | Path,
    *,
    text_store: VectorStore,
    caption_store: CaptionVectorStore | None = None,
    image_store: ImageStore | None = None,
    analyzer: Any | None = None,
    data_root: Path | None = None,
    store_root: Path | None = None,
    chunk_strategy: str = "recursive",
    persist: bool = True,
    extract_visuals: bool = True,
) -> IngestionResult:
    """Ingest one PDF into every configured production retrieval index.

    The text index is built and persisted first and unconditionally: it is the
    only stream the query path requires, so visual work can never cost a
    document its text retrieval.

    Args:
        pdf_path: the PDF to ingest.
        text_store: required; receives the document's chunks.
        caption_store: optional caption index. Skipped when None.
        image_store: optional CLIP index. Skipped when None.
        analyzer: VisualAnalyzer (or any object with ``analyze_evidence``). When
            None, figures are still indexed but carry only their nearby-text
            fallback — no captions are invented.
        data_root: root for figure assets (default: ``data``).
        store_root: root for persisted indices (default: ``settings.vector_store_path``).
        persist: write every touched index to disk.
        extract_visuals: set False to ingest text only.

    Returns:
        IngestionResult describing what was produced, including per-figure
        failures and degraded stages. Never raises for a figure-level problem.
    """
    from mrta.core.config import settings
    from mrta.ingestion.chunker import chunk_pdf
    from mrta.ingestion.pdf_loader import load_pdf

    pdf_path = Path(pdf_path)
    data_root = Path(data_root) if data_root is not None else Path("data")
    store_root = Path(store_root) if store_root is not None else Path(settings.vector_store_path)

    with trace_span("mrta.ingestion.index_document") as span:
        span.set_attribute("ingestion.source", pdf_path.name)

        # --- text stream (required) ---
        pdf = load_pdf(pdf_path)
        chunks: list[Chunk] = chunk_pdf(pdf, strategy=chunk_strategy)
        text_store.add(chunks)
        if persist:
            text_store.save(store_root / TEXT_INDEX_NAME)

        result = IngestionResult(
            doc_id=pdf.doc_id,
            source=pdf.source,
            n_pages=pdf.n_pages,
            n_chunks=len(chunks),
        )

        if not extract_visuals or (caption_store is None and image_store is None):
            span.set_attribute("ingestion.visual_streams", "skipped")
            span.set_attribute("ingestion.chunks", len(chunks))
            return result

        # --- figure extraction (optional, isolated) ---
        figures: list[FigureRecord] = []
        try:
            from mrta.ingestion.figure_extractor import extract_figures

            figures = extract_figures(pdf_path)
        except Exception as exc:  # noqa: BLE001 — a document without figures is still useful
            result.degraded["figure_extraction"] = f"{type(exc).__name__}: {exc}"

        result.n_figures = len(figures)
        if not figures:
            # No figures is a normal outcome, not a failure: the document is
            # ingested and text retrieval works.
            span.set_attribute("ingestion.figures", 0)
            return result

        figures_dir = data_root / FIGURES_SUBDIR
        figures_dir.mkdir(parents=True, exist_ok=True)

        caption_records: list[EvidenceRecord] = []
        visual_records: list[VisualRecord] = []

        for record in figures:
            label = f"{record.doc_id}_p{record.page}_f{record.figure_index}"
            try:
                asset = figure_asset_path(record, figures_dir)
                asset.write_bytes(record.image_bytes)
                # Relative so the stored path stays portable and inside `data/`.
                image_path = str(asset.as_posix())

                if caption_store is not None:
                    evidence, caption_source = build_caption_evidence(record, image_path, analyzer)
                    caption_records.append(evidence)
                    result.caption_sources[caption_source] = (
                        result.caption_sources.get(caption_source, 0) + 1
                    )

                if image_store is not None:
                    visual_records.append(build_visual_record(record, image_path))
            except Exception as exc:  # noqa: BLE001 — isolate one bad figure
                result.figure_failures.append(f"{label}: {type(exc).__name__}: {exc}")

        # --- caption index ---
        if caption_store is not None and caption_records:
            try:
                caption_store.add(caption_records)
                result.n_caption_records = len(caption_records)
                if persist:
                    caption_store.save(store_root / CAPTION_INDEX_NAME)
            except Exception as exc:  # noqa: BLE001 — text index already safe
                result.degraded["caption_index"] = f"{type(exc).__name__}: {exc}"
                result.n_caption_records = 0

        # --- CLIP index ---
        if image_store is not None and visual_records:
            try:
                image_store.add_images(visual_records)
                result.n_visual_records = len(visual_records)
                if persist:
                    image_store.save(store_root / CLIP_INDEX_NAME)
            except Exception as exc:  # noqa: BLE001 — text index already safe
                result.degraded["clip_index"] = f"{type(exc).__name__}: {exc}"
                result.n_visual_records = 0

        span.set_attribute("ingestion.chunks", result.n_chunks)
        span.set_attribute("ingestion.figures", result.n_figures)
        span.set_attribute("ingestion.caption_records", result.n_caption_records)
        span.set_attribute("ingestion.visual_records", result.n_visual_records)
        span.set_attribute("ingestion.figure_failures", len(result.figure_failures))

    return result
