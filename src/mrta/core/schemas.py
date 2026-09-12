"""Shared Pydantic models used across mrta modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from PIL import Image


class PageRecord(BaseModel):
    doc_id: str
    page: int
    text: str
    n_images: int
    source: str
    blocks: list[tuple[float, float, float, float, str, int, int]] = []


class PdfDocument(BaseModel):
    doc_id: str
    source: str
    title: str | None
    n_pages: int
    pages: list[PageRecord]


class Chunk(BaseModel):
    chunk_id: str  # "{doc_id}_p{page}_c{idx}"
    doc_id: str
    source: str
    page: int
    text: str
    section: str | None = None
    n_tokens: int | None = None


class FigureRecord(BaseModel):
    doc_id: str
    source: str
    page: int
    figure_index: int  # 1-indexed per page
    image_bytes: bytes
    # Extended metadata — populated by extract_figures(); absent in older records
    width: int | None = None
    height: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) in points
    nearby_text: str | None = None

    def to_pil(self) -> Image.Image:
        import io

        from PIL import Image

        return Image.open(io.BytesIO(self.image_bytes))

    def to_evidence_record(self) -> EvidenceRecord:
        """Convert to a modality-aware EvidenceRecord for downstream retrieval."""
        eid = f"{self.doc_id}_p{self.page}_f{self.figure_index}"
        return EvidenceRecord(
            evidence_id=eid,
            doc_id=self.doc_id,
            source=self.source,
            page=self.page,
            modality="image",
            figure_index=self.figure_index,
            image_bytes=self.image_bytes,
            bbox=self.bbox,
            nearby_text=self.nearby_text,
        )


class VisualRecord(BaseModel):
    """Canonical identity for one figure image in a CLIP visual index.

    Deliberately narrow: it carries only what a visual retrieval index needs —
    canonical identity plus a path to the image artifact. Unlike EvidenceRecord,
    figure_id is a first-class persisted field, because the visual index must
    round-trip canonical identity without re-consulting the evaluation manifest.

    No image bytes are stored. image_path references the extracted PNG so the
    original image can be loaded lazily when downstream generation needs it.
    """

    document_id: str
    page: int
    figure_id: str
    figure_index: int  # 1-indexed per page
    image_path: str

    # How the image artifact was obtained (PR6). Optional with a default so
    # indices persisted before this field existed still load unchanged.
    extraction_method: str | None = None  # raster_crop | page_render_fallback

    # Retrieval score — set during search; not a persistent field
    retrieval_score: float | None = Field(default=None, exclude=True)

    @property
    def record_id(self) -> str:
        """Stable identifier: '{document_id}_p{page}_f{figure_index}'."""
        return f"{self.document_id}_p{self.page}_f{self.figure_index}"

    def to_pil(self) -> Image.Image:
        """Load the referenced image from disk. Raises FileNotFoundError if missing."""
        from pathlib import Path

        from PIL import Image

        p = Path(self.image_path)
        if not p.exists():
            raise FileNotFoundError(
                f"VisualRecord {self.record_id!r} image_path does not exist: {p}"
            )
        return Image.open(p)


class EvidenceRecord(BaseModel):
    """Modality-aware evidence unit for multimodal RAG retrieval and citation."""

    evidence_id: str  # stable: "{doc_id}_p{page}_f{figure_index}" or "{chunk_id}"
    doc_id: str
    source: str
    page: int
    modality: Literal["text", "image", "page"]

    # Textual content (text chunks) or textual representation (captions/descriptions)
    text: str | None = None

    # Image-specific fields
    figure_index: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) in points
    image_bytes: bytes | None = None

    # VLM-generated semantic fields — populated by VisualAnalyzer
    caption: str | None = None
    detailed_description: str | None = None
    nearby_text: str | None = None
    visual_type: str | None = None

    # Ingestion provenance (PR6). Recorded per record rather than only tallied,
    # so the question "did this figure get a real VLM caption, or only nearby
    # page text?" survives persistence — PR5 measured that distinction as the
    # dominant factor in figure retrieval quality.
    caption_source: str | None = None  # vlm_generated | nearby_text_fallback | empty
    extraction_method: str | None = None  # raster_crop | page_render_fallback

    # Stable filesystem path to the original image — set by ingestion/index-build scripts.
    # Allows the caption index to omit image_bytes at persistence time while still
    # supporting lazy image loading for downstream multimodal generation.
    image_path: str | None = None

    # Retrieval score — set during search; not a persistent field
    retrieval_score: float | None = Field(default=None, exclude=True)

    def retrieval_text(self) -> str:
        """Best available text for embedding — caption > detailed_description > nearby_text."""
        return self.caption or self.detailed_description or self.nearby_text or ""

    def to_pil(self) -> Image.Image:
        """Convert image_bytes to a PIL Image. Raises ValueError if no bytes are stored."""
        import io

        from PIL import Image

        if self.image_bytes is None:
            raise ValueError(f"EvidenceRecord {self.evidence_id!r} has no image_bytes")
        return Image.open(io.BytesIO(self.image_bytes))

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> EvidenceRecord:
        """Wrap a text Chunk as an EvidenceRecord."""
        return cls(
            evidence_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source=chunk.source,
            page=chunk.page,
            modality="text",
            text=chunk.text,
        )


class MultimodalCitation(BaseModel):
    """Structured provenance for one piece of evidence in a multimodal answer.

    The first six fields are the original Stage-7 contract and are always
    populated. The fields below them are additive (PR6): they carry canonical
    retrieval provenance for answers produced by the canonical pipeline and
    stay None on the legacy path, so existing consumers are unaffected.
    """

    label: str  # "[T1]" for text, "[V1]" for visual
    evidence_id: str
    modality: Literal["text", "image", "page"]
    source: str
    page: int
    figure_index: int | None = None

    # --- additive canonical provenance (PR6) ---
    evidence_type: Literal["text", "figure"] | None = None
    document_id: str | None = None
    figure_id: str | None = None
    chunk_id: str | None = None
    image_path: str | None = None
    caption: str | None = None

    # Ranking provenance — internal diagnostics, not a public scoring contract.
    modality_sources: list[str] = Field(default_factory=list)
    rrf_rank: int | None = None
    reranker_rank: int | None = None


class MultimodalAnswer(BaseModel):
    """Return type for MultimodalRAG.ask()."""

    answer: str
    text_citations: list[MultimodalCitation]
    visual_citations: list[MultimodalCitation]
    retrieval_mode: Literal["multimodal", "text_only"] = "multimodal"
    latency_s: float


class EvalReport(BaseModel):
    """Aggregated evaluation results over a benchmark question set."""

    n_questions: int
    answer_relevance: float
    faithfulness: float
    citation_correctness: float
    hallucination_rate: float
    mean_latency_s: float
