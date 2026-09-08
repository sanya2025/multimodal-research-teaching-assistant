"""mrta.retrieval.caption_store — FAISS-backed store for EvidenceRecord caption retrieval.

Indexes EvidenceRecord objects by embedding their retrieval_text() (caption →
detailed_description → nearby_text) using the same Embedder and IndexFlatIP as
VectorStore. This keeps caption-based visual retrieval in the same embedding space
as text chunks without mixing them into one index — fusion happens at the ranking
level, not here.

This store is intentionally separate from VectorStore because EvidenceRecord carries
modality metadata (image, page) that Chunk does not and the return type differs.

Persistence: save() strips image_bytes before writing metadata.jsonl — the caption
index is a text-retrieval artifact; image_path on each record provides a stable
reference for lazy loading by downstream multimodal generation.
"""

from __future__ import annotations

import json
from pathlib import Path
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

    def save(self, path: Path | str) -> None:
        """Write index.faiss + metadata.jsonl + config.json to path.

        image_bytes is stripped before serialization — the caption index stores
        text for retrieval only; image_path on each record provides the reference
        for lazy image loading.
        """
        import faiss

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._ensure_index(), str(p / "index.faiss"))
        lines = []
        for r in self._records:
            stripped = r.model_copy(update={"image_bytes": None})
            lines.append(stripped.model_dump_json())
        (p / "metadata.jsonl").write_text("\n".join(lines), encoding="utf-8")
        (p / "config.json").write_text(
            json.dumps({"dim": self._embedder.dim, "model": self._embedder.model_name}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str, embedder: Embedder) -> CaptionVectorStore:
        """Reload a persisted store. embedder must match the one used at save time.

        Loaded records have image_bytes=None. Use image_path to load the original
        image lazily when needed by downstream multimodal generation.
        """
        import faiss

        from mrta.core.exceptions import RetrievalError

        p = Path(path)
        store = cls(embedder)
        try:
            store._index = faiss.read_index(str(p / "index.faiss"))
        except Exception as e:
            raise RetrievalError(f"Cannot load FAISS index from {p}: {e}") from e
        lines = (p / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        store._records = [EvidenceRecord.model_validate_json(line) for line in lines if line]
        return store
