"""mrta.retrieval.vector_store — FAISS-backed vector store with Chunk metadata."""

from __future__ import annotations

import json
from pathlib import Path

from mrta.core.exceptions import RetrievalError
from mrta.core.schemas import Chunk
from mrta.retrieval.embedder import Embedder


class VectorStore:
    """IndexFlatIP FAISS index paired with a parallel list of Chunks.

    Uses inner product on L2-normalized vectors — mathematically equivalent to
    cosine similarity, but requires no FAISS training at this scale.
    This is the swap boundary for Qdrant (ADR-002): replace this class with a
    Qdrant implementation that exposes the same four-method interface.
    """

    def __init__(self, embedder: Embedder) -> None:
        import faiss

        self._embedder = embedder
        self._index = faiss.IndexFlatIP(embedder.dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Embed chunks and add them to the index."""
        if not chunks:
            return
        embs = self._embedder.embed([c.text for c in chunks])
        self._index.add(embs)
        self._chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return top-k Chunks by cosine similarity to query."""
        q = self._embedder.embed([query])
        scores, idx = self._index.search(q, k)
        return [self._chunks[i] for i in idx[0] if 0 <= i < len(self._chunks)]

    def save(self, path: Path | str) -> None:
        """Write index.faiss + metadata.jsonl + config.json to path."""
        import faiss

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "index.faiss"))
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
