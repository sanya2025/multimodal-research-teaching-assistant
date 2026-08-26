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
    """Structured provenance for one piece of evidence in a multimodal answer."""

    label: str  # "[T1]" for text, "[V1]" for visual
    evidence_id: str
    modality: Literal["text", "image", "page"]
    source: str
    page: int
    figure_index: int | None = None


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
