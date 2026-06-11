"""mrta.retrieval.reranker — Cross-encoder reranking for RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mrta.core.schemas import Chunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker:
    """Re-scores retrieved chunks using a cross-encoder model.

    Loads the cross-encoder lazily on first instantiation. The model is
    downloaded from HuggingFace Hub on first use — mock in tests to avoid
    network access.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model: CrossEncoder = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 3) -> list[Chunk]:
        """Return top_n chunks sorted by cross-encoder relevance score (descending).

        If top_n exceeds len(chunks), all chunks are returned in score order.
        """
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_n]]
