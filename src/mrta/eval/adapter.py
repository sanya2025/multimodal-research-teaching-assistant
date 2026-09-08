"""mrta.eval.adapter — maps production retrieval output to CanonicalEvidence.

This is evaluation-only code. It does not change production retrieval behavior.

Mapping rules:
  - Chunk.source (filename) → document_id via manifest source→doc-id lookup.
  - Chunk.page → page_number (already 1-indexed physical PDF page).
  - Text chunks always have figure_id=None.
  - EvidenceRecord with modality "image"/"page" → figure_id via
    (source, page, figure_index) lookup in the manifest figure map.
    If the figure_index is not in the manifest, figure_id defaults to None.

Document ID fallback: if Chunk.source is not in the manifest, the adapter
uses the raw source string as document_id so metrics can still be computed
(they will simply not match any manifest target).
"""

from __future__ import annotations

from mrta.core.schemas import Chunk, EvidenceRecord, VisualRecord
from mrta.eval.types import CanonicalEvidence, RetrievedCandidate


class EvalAdapter:
    """Maps Chunk / EvidenceRecord → RetrievedCandidate with canonical evidence."""

    def __init__(self, manifest: dict) -> None:
        self._source_to_docid: dict[str, str] = {
            doc["filename"]: doc["document_id"] for doc in manifest["documents"]
        }
        # (source, page_number, figure_index) → figure_id
        self._figure_map: dict[tuple[str, int, int], str] = {}
        for doc in manifest["documents"]:
            src = doc["filename"]
            for fig in doc.get("figures", []):
                key = (src, fig["page_number"], fig["figure_index"])
                self._figure_map[key] = fig["figure_id"]

    def chunk_to_candidate(
        self,
        chunk: Chunk,
        score: float,
        rank: int,
    ) -> RetrievedCandidate:
        """Wrap a text Chunk as a RetrievedCandidate (figure_id always None)."""
        doc_id = self._source_to_docid.get(chunk.source, chunk.source)
        return RetrievedCandidate(
            candidate_id=chunk.chunk_id,
            evidence=CanonicalEvidence(
                document_id=doc_id,
                page_number=chunk.page,
                figure_id=None,
            ),
            score=score,
            rank=rank,
        )

    def from_caption_record(
        self,
        record: EvidenceRecord,
        score: float,
        rank: int,
    ) -> RetrievedCandidate:
        """Map a CaptionVectorStore result to a RetrievedCandidate.

        Named alias for evidence_record_to_candidate — makes caption-retrieval
        call sites self-documenting. Uses identical canonical evidence semantics.
        """
        return self.evidence_record_to_candidate(record, score, rank)

    def from_visual_record(
        self,
        record: VisualRecord,
        score: float,
        rank: int,
    ) -> RetrievedCandidate:
        """Map an ImageStore (CLIP) result to a RetrievedCandidate.

        VisualRecord already carries its canonical figure_id, resolved when the
        index was built, so no manifest lookup is needed here. Canonical evidence
        semantics are identical to the text and caption paths.
        """
        return RetrievedCandidate(
            candidate_id=record.record_id,
            evidence=CanonicalEvidence(
                document_id=record.document_id,
                page_number=record.page,
                figure_id=record.figure_id,
            ),
            score=score,
            rank=rank,
        )

    def evidence_record_to_candidate(
        self,
        record: EvidenceRecord,
        score: float,
        rank: int,
    ) -> RetrievedCandidate:
        """Wrap an EvidenceRecord as a RetrievedCandidate.

        For image/page modalities, resolves figure_id from the manifest.
        For text modality, figure_id is always None.
        """
        doc_id = self._source_to_docid.get(record.source, record.source)
        figure_id: str | None = None
        if record.modality in ("image", "page") and record.figure_index is not None:
            key = (record.source, record.page, record.figure_index)
            figure_id = self._figure_map.get(key)
        return RetrievedCandidate(
            candidate_id=record.evidence_id,
            evidence=CanonicalEvidence(
                document_id=doc_id,
                page_number=record.page,
                figure_id=figure_id,
            ),
            score=score,
            rank=rank,
        )
