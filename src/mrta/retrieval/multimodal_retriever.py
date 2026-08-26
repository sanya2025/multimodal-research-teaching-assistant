"""mrta.retrieval.multimodal_retriever — Unified text + visual retrieval with RRF.

Combines up to three independently-ranked evidence streams:

    text      — VectorStore (sentence-transformer, 384-dim)
    caption   — CaptionVectorStore (sentence-transformer, 384-dim, from VLM descriptions)
    visual    — VisualVectorStore (CLIP, 512-dim, from image embeddings)

Their rankings are fused via Reciprocal Rank Fusion (fusion.py). Raw cosine
scores from different embedding spaces are never compared directly; only rank
order feeds RRF.

Optional reranking: a cross-encoder Reranker can re-score the fused candidates
using each record's textual serialization (retrieval_text()). This improves
text-rich evidence but cannot evaluate pure visual evidence independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mrta.core.schemas import EvidenceRecord
from mrta.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from mrta.retrieval.vector_store import VectorStore

if TYPE_CHECKING:
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.reranker import Reranker
    from mrta.retrieval.visual_vector_store import VisualVectorStore


class MultimodalRetriever:
    """Retrieves evidence from text, caption, and visual stores and fuses via RRF.

    At minimum, a VectorStore is required. CaptionVectorStore and
    VisualVectorStore are optional — the retriever degrades gracefully to
    text-only when they are absent or empty.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        caption_store: CaptionVectorStore | None = None,
        visual_store: VisualVectorStore | None = None,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        reranker_top_n: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._caption_store = caption_store
        self._visual_store = visual_store
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._reranker_top_n = reranker_top_n

    def retrieve(
        self,
        query: str,
        k_text: int = 5,
        k_visual: int = 5,
        k_final: int = 8,
    ) -> list[EvidenceRecord]:
        """Return top k_final EvidenceRecords fused from all available stores.

        Each returned record has retrieval_score set to its RRF score.
        """
        fused = self.retrieve_with_fusion_details(query, k_text, k_visual, k_final)
        return [fr.record.model_copy(update={"retrieval_score": fr.rrf_score}) for fr in fused]

    def retrieve_with_fusion_details(
        self,
        query: str,
        k_text: int = 5,
        k_visual: int = 5,
        k_final: int = 8,
    ) -> list[FusedResult]:
        """Same as retrieve() but exposes full RRF diagnostics per result.

        Useful for notebooks, debugging, and evaluation.
        """
        named_lists: dict[str, list[EvidenceRecord]] = {}

        text_results = self._vector_store.search_with_scores(query, k=k_text)
        named_lists["text"] = [EvidenceRecord.from_chunk(chunk) for chunk, _ in text_results]

        if self._caption_store is not None:
            named_lists["caption"] = self._caption_store.search(query, k=k_visual)

        if self._visual_store is not None:
            named_lists["visual"] = self._visual_store.search(query, k=k_visual)

        fused = reciprocal_rank_fusion(named_lists, k=self._rrf_k, top_n=k_final)

        if self._reranker is not None:
            fused = self._apply_reranker(query, fused)

        return fused

    def _apply_reranker(
        self,
        query: str,
        fused: list[FusedResult],
    ) -> list[FusedResult]:
        """Re-score fused evidence using the cross-encoder on textual serializations.

        Limitation: visual evidence without caption/description text is represented
        as an empty string. Full cross-modal reranking is a future milestone.
        """
        if not fused:
            return fused
        candidates = fused[: self._reranker_top_n]
        pairs = [(query, fr.record.retrieval_text()) for fr in candidates]
        scores = self._reranker._model.predict(pairs)  # type: ignore[union-attr]
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [fr for _, fr in ranked]
