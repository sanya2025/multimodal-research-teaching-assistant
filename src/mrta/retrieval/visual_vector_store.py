"""mrta.retrieval.visual_vector_store — FAISS-backed store for CLIP image retrieval.

Indexes EvidenceRecord objects by embedding their image_bytes with CLIPEmbedder.
Text queries are embedded with embed_text() — the same CLIP space — enabling
direct text-to-image retrieval without any VLM-generated caption.

This store is intentionally separate from VectorStore (sentence-transformer, 384-dim)
and CaptionVectorStore (sentence-transformer, 384-dim). CLIP vectors (512-dim) live in
a different space and must not be mixed into those indexes. Fusion across stores happens
at the ranking level via RRF (Stage 4), not here.

Records without image_bytes are skipped silently on add() — page records may lack bytes
at index time if they were not fully rendered before ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mrta.core.exceptions import RetrievalError
from mrta.core.schemas import EvidenceRecord
from mrta.multimodal.clip_embedder import CLIPEmbedder

if TYPE_CHECKING:
    import faiss


class VisualVectorStore:
    """IndexFlatIP FAISS index over CLIP image embeddings.

    Images are embedded with CLIPEmbedder.embed_image(); text queries
    are embedded with CLIPEmbedder.embed_text(). Both live in the same
    CLIP representation space, so dot-product is cosine similarity.
    """

    def __init__(self, clip: CLIPEmbedder) -> None:
        self._clip = clip
        self._index: faiss.Index | None = None
        self._records: list[EvidenceRecord] = []

    def _ensure_index(self) -> faiss.Index:
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self._clip.dim)
        return self._index

    @property
    def size(self) -> int:
        """Number of records currently indexed."""
        return len(self._records)

    def add(self, records: list[EvidenceRecord]) -> None:
        """Embed images and add to the index. Records without image_bytes are skipped."""
        if not records:
            return
        vecs: list[np.ndarray] = []
        indexable: list[EvidenceRecord] = []
        for rec in records:
            if rec.image_bytes is None:
                continue
            vecs.append(self._clip.embed_image(rec.to_pil()))
            indexable.append(rec)
        if not vecs:
            return
        arr = np.stack(vecs).astype("float32")
        self._ensure_index().add(arr)
        self._records.extend(indexable)

    def search(self, query: str, k: int = 5) -> list[EvidenceRecord]:
        """Return top-k EvidenceRecords by CLIP cosine similarity to query text.

        Each returned record is a copy with retrieval_score set.
        """
        return [rec for rec, _ in self.search_with_scores(query, k)]

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[EvidenceRecord, float]]:
        """Return top-k (EvidenceRecord, cosine_score) pairs, deduplicated by evidence_id.

        Each returned record is a copy with retrieval_score set.
        """
        if not self._records:
            return []
        q = self._clip.embed_text(query).reshape(1, -1).astype("float32")
        fetch_k = min(k * 2, len(self._records))
        scores, idx = self._ensure_index().search(q, fetch_k)
        seen: set[str] = set()
        results: list[tuple[EvidenceRecord, float]] = []
        for rank, i in enumerate(idx[0]):
            if not (0 <= i < len(self._records)):
                continue
            rec = self._records[i]
            if rec.evidence_id in seen:
                continue
            seen.add(rec.evidence_id)
            score = float(scores[0][rank])
            results.append((rec.model_copy(update={"retrieval_score": score}), score))
            if len(results) == k:
                break
        return results

    def save(self, path: Path | str) -> None:
        """Write index.faiss + metadata.jsonl + config.json to path."""
        import faiss

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._ensure_index(), str(p / "index.faiss"))
        # image_bytes are excluded: they can be MB each and are re-fetchable from the
        # source PDF. The FAISS index preserves retrieval; metadata preserves provenance.
        (p / "metadata.jsonl").write_text(
            "\n".join(r.model_dump_json(exclude={"image_bytes"}) for r in self._records),
            encoding="utf-8",
        )
        (p / "config.json").write_text(
            json.dumps({"dim": self._clip.dim, "model": self._clip.model_name}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str, clip: CLIPEmbedder) -> VisualVectorStore:
        """Reload a persisted store. clip must match the one used at save time."""
        import faiss

        p = Path(path)
        store = cls(clip)
        try:
            store._index = faiss.read_index(str(p / "index.faiss"))
        except Exception as e:
            raise RetrievalError(f"Cannot load FAISS index from {p}: {e}") from e
        lines = (p / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        store._records = [EvidenceRecord.model_validate_json(line) for line in lines if line]
        return store
