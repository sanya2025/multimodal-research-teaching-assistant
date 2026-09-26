"""mrta.retrieval.image_store_adapter — read the canonical CLIP index from the legacy path.

Why this exists
---------------
Two CLIP-backed visual stores coexist in this repository:

``VisualVectorStore``
    Indexes :class:`~mrta.core.schemas.EvidenceRecord` and is what
    :class:`~mrta.retrieval.multimodal_retriever.MultimodalRetriever` was written
    against. Nothing in the production ingestion path ever writes it.

``ImageStore``
    Indexes :class:`~mrta.core.schemas.VisualRecord` and *is* what
    :func:`mrta.ingestion.document_indexer.index_document` persists.

The two record types are not interchangeable: ``VisualRecord`` is deliberately
narrower and carries no ``source`` (the PDF filename), because canonical identity
is ``document_id`` there. Pointing ``VisualVectorStore.load()`` at the persisted
CLIP index therefore fails schema validation.

This adapter bridges them, so the legacy retrieval path reads the same CLIP index
production ingestion writes instead of a second index nothing populates. Only the
search surface ``MultimodalRetriever`` actually calls is implemented; this is an
adapter, not a store.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mrta.core.schemas import EvidenceRecord, VisualRecord

if TYPE_CHECKING:
    from mrta.retrieval.image_store import ImageStore

# Resolves a document_id to its PDF filename. Returning None is normal for a
# document whose text chunks are not loaded, and yields a graceful fallback.
SourceResolver = Callable[[str], str | None]


def visual_record_to_evidence(record: VisualRecord, source: str | None = None) -> EvidenceRecord:
    """Adapt one CLIP-index record to the EvidenceRecord the legacy path expects.

    ``evidence_id`` uses :attr:`VisualRecord.record_id`, which is the same
    ``{doc_id}_p{page}_f{figure_index}`` form the caption index assigns. That
    match is what lets RRF recognise a figure found by both the caption stream
    and the CLIP stream as one piece of evidence rather than two.

    ``source`` falls back to ``document_id`` when unresolved: a citation that
    names the document id is degraded but still truthful, whereas inventing a
    filename would not be.
    """
    return EvidenceRecord(
        evidence_id=record.record_id,
        doc_id=record.document_id,
        source=source or record.document_id,
        page=record.page,
        modality="image",
        figure_index=record.figure_index,
        image_path=record.image_path,
        extraction_method=record.extraction_method,
        retrieval_score=record.retrieval_score,
    )


class ImageStoreAdapter:
    """Presents an :class:`ImageStore` through the ``VisualVectorStore`` search API.

    Deliberately read-only. Indexing stays with ``ImageStore`` so there is exactly
    one CLIP index in the system and one place that writes it.
    """

    def __init__(
        self,
        image_store: ImageStore,
        source_resolver: SourceResolver | None = None,
    ) -> None:
        """
        Args:
            image_store: the loaded canonical CLIP index.
            source_resolver: maps document_id -> PDF filename. Resolved per call
                rather than snapshotted, so documents uploaded after startup
                still cite their filename.
        """
        self._image_store = image_store
        self._source_resolver = source_resolver

    @property
    def size(self) -> int:
        """Number of records in the underlying CLIP index."""
        return self._image_store.size

    def _resolve_source(self, document_id: str) -> str | None:
        if self._source_resolver is None:
            return None
        try:
            return self._source_resolver(document_id)
        except Exception:  # noqa: BLE001
            # Source resolution is cosmetic; never let it break retrieval.
            return None

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[EvidenceRecord, float]]:
        """Return up to k (EvidenceRecord, cosine_score) pairs, highest score first."""
        hits = self._image_store.search(query, top_k=k)
        return [
            (visual_record_to_evidence(rec, self._resolve_source(rec.document_id)), score)
            for rec, score in hits
        ]

    def search(self, query: str, k: int = 5) -> list[EvidenceRecord]:
        """Return up to k EvidenceRecords by CLIP similarity to the query text."""
        return [rec for rec, _ in self.search_with_scores(query, k)]
