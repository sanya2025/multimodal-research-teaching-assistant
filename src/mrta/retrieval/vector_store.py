"""mrta.retrieval.vector_store — FAISS-backed vector store with Chunk metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from mrta.core.exceptions import RetrievalError
from mrta.core.schemas import Chunk
from mrta.retrieval.embedder import Embedder

if TYPE_CHECKING:
    import faiss


class VectorStore:
    """IndexFlatIP FAISS index paired with a parallel list of Chunks.

    Uses inner product on L2-normalized vectors — mathematically equivalent to
    cosine similarity, but requires no FAISS training at this scale.
    This is the swap boundary for Qdrant (ADR-002): replace this class with a
    Qdrant implementation that exposes the same four-method interface.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: faiss.Index | None = None  # created on first add()
        self._chunks: list[Chunk] = []

    def _ensure_index(self) -> faiss.Index:
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self._embedder.dim)
        return self._index

    def add(self, chunks: list[Chunk]) -> None:
        """Embed chunks and add them to the index."""
        if not chunks:
            return
        embs = self._embedder.embed([c.text for c in chunks])
        self._ensure_index().add(embs)
        self._chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return top-k Chunks by cosine similarity to query."""
        return [chunk for chunk, _ in self.search_with_scores(query, k)]

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return top-k unique (Chunk, cosine_score) pairs, deduplicated by chunk_id.

        Over-fetches k*3 candidates from FAISS so that duplicates (e.g. from a
        document indexed more than once) are absorbed before the k-cap is applied.
        """
        if not self._chunks:
            return []
        q = self._embedder.embed([query])
        fetch_k = min(k * 3, len(self._chunks))
        scores, idx = self._ensure_index().search(q, fetch_k)
        seen: set[str] = set()
        results: list[tuple[Chunk, float]] = []
        for rank, i in enumerate(idx[0]):
            if not (0 <= i < len(self._chunks)):
                continue
            chunk = self._chunks[i]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            results.append((chunk, float(scores[0][rank])))
            if len(results) == k:
                break
        return results

    def save(self, path: Path | str) -> None:
        """Write index.faiss + metadata.jsonl + config.json to path."""
        import faiss

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._ensure_index(), str(p / "index.faiss"))
        (p / "metadata.jsonl").write_text(
            "\n".join(c.model_dump_json() for c in self._chunks),
            encoding="utf-8",
        )
        (p / "config.json").write_text(
            json.dumps({"dim": self._embedder.dim, "model": self._embedder.model_name}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str, embedder: Embedder) -> VectorStore:
        """Reload a persisted store. embedder must match the one used at save time."""
        import faiss

        p = Path(path)
        store = cls(embedder)
        try:
            store._index = faiss.read_index(str(p / "index.faiss"))
        except Exception as e:
            raise RetrievalError(f"Cannot load FAISS index from {p}: {e}") from e
        lines = (p / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        store._chunks = [Chunk.model_validate_json(line) for line in lines if line]
        return store
