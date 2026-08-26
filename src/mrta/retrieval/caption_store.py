"""mrta.retrieval.caption_store — FAISS-backed store for EvidenceRecord caption retrieval.

Indexes EvidenceRecord objects by embedding their retrieval_text() (caption →
detailed_description → nearby_text) using the same Embedder and IndexFlatIP as
VectorStore. This keeps caption-based visual retrieval in the same embedding space
as text chunks without mixing them into one index — fusion happens at the ranking
level, not here.

This store is intentionally separate from VectorStore because EvidenceRecord carries
modality metadata (image, page) that Chunk does not and the return type differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mrta.core.schemas import EvidenceRecord
from mrta.retrieval.embedder import Embedder

if TYPE_CHECKING:
    import faiss


class CaptionVectorStore:
    """IndexFlatIP FAISS index over EvidenceRecord caption text.

    Retrieval uses the record's best available text: caption, then
    detailed_description, then nearby_text. Records with no text at all are
    embedded as empty strings and will score near zero for most queries.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._records: list[EvidenceRecord] = []

    def _ensure_index(self) -> faiss.Index:
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self._embedder.dim)
        return self._index

    @property
    def size(self) -> int:
        """Number of records currently indexed."""
        return len(self._records)

    def add(self, records: list[EvidenceRecord]) -> None:
        """Embed records by their retrieval_text() and add them to the index."""
        if not records:
            return
        texts = [r.retrieval_text() for r in records]
        embs = self._embedder.embed(texts)
        self._ensure_index().add(embs)
        self._records.extend(records)

    def search(self, query: str, k: int = 5) -> list[EvidenceRecord]:
        """Return top-k EvidenceRecords by cosine similarity to query.

        Each returned record is a copy with retrieval_score set.
        The internally stored records are not mutated.
        """
        return [record for record, _ in self.search_with_scores(query, k)]

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[EvidenceRecord, float]]:
        """Return top-k (EvidenceRecord, cosine_score) pairs, deduplicated by evidence_id.

        Each returned record is a copy with retrieval_score set.
        """
        if not self._records:
            return []
        q = self._embedder.embed([query])
        fetch_k = min(k * 2, len(self._records))
        scores, idx = self._ensure_index().search(q, fetch_k)
        seen: set[str] = set()
        results: list[tuple[EvidenceRecord, float]] = []
        for rank, i in enumerate(idx[0]):
            if not (0 <= i < len(self._records)):
                continue
            record = self._records[i]
            if record.evidence_id in seen:
                continue
            seen.add(record.evidence_id)
            score = float(scores[0][rank])
            results.append((record.model_copy(update={"retrieval_score": score}), score))
            if len(results) == k:
                break
        return results
