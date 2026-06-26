"""Unit tests for Reranker and rag_query reranking integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mrta.core.rag_pipeline import rag_query
from mrta.core.schemas import Chunk

FAKE_CHUNKS = [
    Chunk(chunk_id="d_p1_c0", doc_id="d", source="a.pdf", page=1, text="Attention is fundamental."),
    Chunk(chunk_id="d_p2_c0", doc_id="d", source="a.pdf", page=2, text="Transformers changed NLP."),
    Chunk(
        chunk_id="d_p3_c0",
        doc_id="d",
        source="a.pdf",
        page=3,
        text="BERT uses bidirectional encoding.",
    ),
]


class TestReranker:
    def _make_reranker(self, scores: list[float]):  # type: ignore[no-untyped-def]
        from mrta.retrieval.reranker import Reranker

        with patch("sentence_transformers.CrossEncoder") as mock_ce_cls:
            mock_ce = MagicMock()
            mock_ce.predict.return_value = scores
            mock_ce_cls.return_value = mock_ce
            reranker = Reranker()
        return reranker, mock_ce

    def test_rerank_returns_top_n(self) -> None:
        reranker, _ = self._make_reranker([0.9, 0.3, 0.6])
        result = reranker.rerank("query", FAKE_CHUNKS, top_n=2)
        assert len(result) == 2

    def test_rerank_orders_by_score_descending(self) -> None:
        reranker, _ = self._make_reranker([0.2, 0.9, 0.5])
        result = reranker.rerank("query", FAKE_CHUNKS, top_n=3)
        assert result[0] == FAKE_CHUNKS[1]  # score 0.9
        assert result[1] == FAKE_CHUNKS[2]  # score 0.5
        assert result[2] == FAKE_CHUNKS[0]  # score 0.2

    def test_rerank_empty_input(self) -> None:
        reranker, _ = self._make_reranker([])
        result = reranker.rerank("query", [], top_n=3)
        assert result == []

    def test_rerank_top_n_exceeds_chunks(self) -> None:
        reranker, _ = self._make_reranker([0.4, 0.8])
        result = reranker.rerank("query", FAKE_CHUNKS[:2], top_n=10)
        assert len(result) == 2

    def test_rerank_calls_predict_with_pairs(self) -> None:
        reranker, mock_ce = self._make_reranker([0.5, 0.7, 0.3])
        reranker.rerank("my query", FAKE_CHUNKS, top_n=2)
        call_args = mock_ce.predict.call_args[0][0]
        assert call_args == [("my query", c.text) for c in FAKE_CHUNKS]


class TestRagPipelineReranking:
    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock()
        store.search_with_scores.return_value = [(c, 0.9) for c in FAKE_CHUNKS]
        return store

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        llm = MagicMock()
        llm.chat.return_value = "answer"
        return llm

    @pytest.fixture
    def mock_reranker(self) -> MagicMock:
        reranker = MagicMock()
        reranker.rerank.return_value = FAKE_CHUNKS[:2]
        return reranker

    def test_rag_query_calls_reranker_when_provided(
        self, mock_store: MagicMock, mock_llm: MagicMock, mock_reranker: MagicMock
    ) -> None:
        rag_query("test?", mock_store, mock_llm, reranker=mock_reranker, rerank_top_n=2)
        mock_reranker.rerank.assert_called_once()

    def test_rag_query_skips_reranker_when_none(
        self, mock_store: MagicMock, mock_llm: MagicMock
    ) -> None:
        result = rag_query("test?", mock_store, mock_llm, reranker=None)
        assert result["sources"] == FAKE_CHUNKS

    def test_rag_query_passes_reranked_chunks_to_prompt(
        self, mock_store: MagicMock, mock_llm: MagicMock, mock_reranker: MagicMock
    ) -> None:
        result = rag_query("test?", mock_store, mock_llm, reranker=mock_reranker, rerank_top_n=2)
        assert result["sources"] == FAKE_CHUNKS[:2]
