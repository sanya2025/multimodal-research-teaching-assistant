"""Unit tests for MultimodalRetriever.

All stores are mocked — no sentence-transformers, CLIP, or FAISS required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mrta.core.schemas import Chunk, EvidenceRecord
from mrta.retrieval.fusion import FusedResult
from mrta.retrieval.multimodal_retriever import MultimodalRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(cid: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc1",
        source="test.pdf",
        page=page,
        text="some text",
    )


def make_evidence(eid: str, modality: str = "image", page: int = 1) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        doc_id="doc1",
        source="test.pdf",
        page=page,
        modality=modality,  # type: ignore[arg-type]
        caption="a figure" if modality == "image" else None,
        text="some text" if modality == "text" else None,
    )


def make_vector_store(chunks: list[Chunk]) -> MagicMock:
    """Mock VectorStore.search_with_scores → list[(chunk, 0.9)]."""
    store = MagicMock()
    store.search_with_scores.return_value = [(c, 0.9 - i * 0.1) for i, c in enumerate(chunks)]
    return store


def make_caption_store(records: list[EvidenceRecord]) -> MagicMock:
    store = MagicMock()
    store.search.return_value = records
    return store


def make_visual_store(records: list[EvidenceRecord]) -> MagicMock:
    store = MagicMock()
    store.search_with_scores.return_value = [(r, 0.9) for r in records]
    return store


# ---------------------------------------------------------------------------
# TestMultimodalRetrieverTextOnly
# ---------------------------------------------------------------------------


class TestMultimodalRetrieverTextOnly:
    def test_text_only_returns_evidence_records(self) -> None:
        chunk = make_chunk("c1")
        vs = make_vector_store([chunk])
        retriever = MultimodalRetriever(vector_store=vs)
        results = retriever.retrieve("query")
        assert len(results) == 1
        assert isinstance(results[0], EvidenceRecord)

    def test_text_only_modality_is_text(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        retriever = MultimodalRetriever(vector_store=vs)
        results = retriever.retrieve("query")
        assert results[0].modality == "text"

    def test_text_chunk_evidence_id_from_chunk_id(self) -> None:
        chunk = make_chunk("chunk_abc")
        vs = make_vector_store([chunk])
        retriever = MultimodalRetriever(vector_store=vs)
        results = retriever.retrieve("query")
        assert results[0].evidence_id == "chunk_abc"

    def test_retrieval_score_set_to_rrf_score(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        retriever = MultimodalRetriever(vector_store=vs)
        results = retriever.retrieve("query")
        assert results[0].retrieval_score is not None
        assert results[0].retrieval_score > 0.0

    def test_k_final_limits_output(self) -> None:
        chunks = [make_chunk(f"c{i}") for i in range(10)]
        vs = make_vector_store(chunks)
        retriever = MultimodalRetriever(vector_store=vs)
        results = retriever.retrieve("query", k_text=10, k_final=3)
        assert len(results) <= 3

    def test_no_caption_or_visual_store(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        retriever = MultimodalRetriever(vector_store=vs)
        results = retriever.retrieve("query")
        vs.search_with_scores.assert_called_once()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# TestMultimodalRetrieverWithCaptionStore
# ---------------------------------------------------------------------------


class TestMultimodalRetrieverWithCaptionStore:
    def test_caption_store_results_included(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        cs = make_caption_store([make_evidence("img1", "image")])
        retriever = MultimodalRetriever(vector_store=vs, caption_store=cs)
        results = retriever.retrieve("query")
        ids = {r.evidence_id for r in results}
        assert "c1" in ids
        assert "img1" in ids

    def test_caption_store_search_called(self) -> None:
        vs = make_vector_store([])
        cs = make_caption_store([])
        retriever = MultimodalRetriever(vector_store=vs, caption_store=cs)
        retriever.retrieve("query about figures")
        cs.search.assert_called_once_with("query about figures", k=5)


# ---------------------------------------------------------------------------
# TestMultimodalRetrieverWithVisualStore
# ---------------------------------------------------------------------------


class TestMultimodalRetrieverWithVisualStore:
    def test_visual_store_results_included(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        vstore = make_visual_store([make_evidence("vis1", "image")])
        retriever = MultimodalRetriever(vector_store=vs, visual_store=vstore)
        results = retriever.retrieve("query")
        ids = {r.evidence_id for r in results}
        assert "vis1" in ids

    def test_visual_store_search_called(self) -> None:
        vs = make_vector_store([])
        vstore = make_visual_store([])
        retriever = MultimodalRetriever(vector_store=vs, visual_store=vstore)
        retriever.retrieve("architecture diagram", k_visual=3)
        vstore.search_with_scores.assert_called_once_with("architecture diagram", k=3)

    def test_empty_visual_store_still_returns_text(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        vstore = make_visual_store([])
        retriever = MultimodalRetriever(vector_store=vs, visual_store=vstore)
        results = retriever.retrieve("query")
        assert len(results) == 1
        assert results[0].evidence_id == "c1"


# ---------------------------------------------------------------------------
# TestMultimodalRetrieverAllStores
# ---------------------------------------------------------------------------


class TestMultimodalRetrieverAllStores:
    def _make_retriever(self) -> tuple[MultimodalRetriever, MagicMock, MagicMock, MagicMock]:
        vs = make_vector_store([make_chunk("c1"), make_chunk("c2")])
        cs = make_caption_store([make_evidence("cap1", "image"), make_evidence("cap2", "image")])
        vstore = make_visual_store([make_evidence("vis1", "image"), make_evidence("c1", "text")])
        retriever = MultimodalRetriever(vector_store=vs, caption_store=cs, visual_store=vstore)
        return retriever, vs, cs, vstore

    def test_all_three_stores_queried(self) -> None:
        retriever, vs, cs, vstore = self._make_retriever()
        retriever.retrieve("query")
        vs.search_with_scores.assert_called_once()
        cs.search.assert_called_once()
        vstore.search_with_scores.assert_called_once()

    def test_deduplication_on_shared_id(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        vstore = make_visual_store([make_evidence("c1", "text")])
        retriever = MultimodalRetriever(vector_store=vs, visual_store=vstore)
        results = retriever.retrieve("query")
        ids = [r.evidence_id for r in results]
        assert ids.count("c1") == 1

    def test_fused_result_list_from_details(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        cs = make_caption_store([make_evidence("img1")])
        retriever = MultimodalRetriever(vector_store=vs, caption_store=cs)
        details = retriever.retrieve_with_fusion_details("query")
        assert all(isinstance(d, FusedResult) for d in details)

    def test_per_list_rank_in_details(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        retriever = MultimodalRetriever(vector_store=vs)
        details = retriever.retrieve_with_fusion_details("query")
        assert "text" in details[0].per_list_rank


# ---------------------------------------------------------------------------
# TestRerankerHook
# ---------------------------------------------------------------------------


class TestRerankerHook:
    def test_reranker_not_called_when_none(self) -> None:
        vs = make_vector_store([make_chunk("c1")])
        retriever = MultimodalRetriever(vector_store=vs, reranker=None)
        result = retriever.retrieve("query")
        assert len(result) == 1

    def test_reranker_applied_to_candidates(self) -> None:
        vs = make_vector_store([make_chunk("c1"), make_chunk("c2")])
        reranker = MagicMock()
        reranker._model.predict.return_value = np.array([0.3, 0.9])
        retriever = MultimodalRetriever(vector_store=vs, reranker=reranker, reranker_top_n=2)
        details = retriever.retrieve_with_fusion_details("query")
        reranker._model.predict.assert_called_once()
        assert len(details) == 2

    def test_reranker_top_n_limits_candidates_passed(self) -> None:
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        vs = make_vector_store(chunks)
        reranker = MagicMock()
        reranker._model.predict.return_value = np.array([0.5, 0.8, 0.3])
        retriever = MultimodalRetriever(vector_store=vs, reranker=reranker, reranker_top_n=3)
        retriever.retrieve_with_fusion_details("query", k_text=5, k_final=5)
        call_args = reranker._model.predict.call_args[0][0]
        assert len(call_args) == 3
