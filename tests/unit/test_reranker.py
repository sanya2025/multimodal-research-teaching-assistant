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


# ---------------------------------------------------------------------------
# PR5 — CrossEncoderReranker over fused multimodal candidates
# ---------------------------------------------------------------------------


def _fused(
    canonical_id: str,
    evidence_type: str = "text",
    payload: dict | None = None,
    rrf_score: float = 0.05,
    document_id: str = "attention_is_all_you_need",
    page: int = 1,
    figure_id: str | None = None,
    modality_sources: tuple[str, ...] = ("text",),
    source_ranks: dict | None = None,
):  # type: ignore[no-untyped-def]
    from mrta.retrieval.fusion import FusedCandidate

    return FusedCandidate(
        canonical_id=canonical_id,
        evidence_type=evidence_type,
        document_id=document_id,
        page=page,
        figure_id=figure_id,
        chunk_id=canonical_id if evidence_type == "text" else None,
        score=rrf_score,
        modality_sources=modality_sources,
        source_ranks=source_ranks or {"text": 1},
        payload=payload or {},
    )


def _text_cand(cid: str, text: str, **kw):  # type: ignore[no-untyped-def]
    return _fused(cid, "text", {"chunk_text": text}, **kw)


def _figure_cand(cid: str, payload: dict, **kw):  # type: ignore[no-untyped-def]
    kw.setdefault("figure_id", "fig_attention_mechanisms")
    kw.setdefault("modality_sources", ("caption", "clip"))
    kw.setdefault("source_ranks", {"caption": 2, "clip": 3})
    return _fused(cid, "figure", payload, **kw)


def _make_ce_reranker(scores):  # type: ignore[no-untyped-def]
    """CrossEncoderReranker with a mocked model returning `scores` from predict."""
    from mrta.retrieval.reranker import CrossEncoderReranker

    with patch("sentence_transformers.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        mock_cls.return_value = mock_model
        reranker = CrossEncoderReranker(model_name="mock-model")
    return reranker, mock_model


class TestCandidateTextRepresentation:
    """What the cross-encoder actually sees for each candidate kind (spec 7, 8)."""

    def test_text_candidate_uses_chunk_text(self) -> None:
        from mrta.retrieval.reranker import candidate_text_source, candidate_to_reranker_text

        c = _text_cand("text|d|c1", "Scaled dot-product attention is computed as ...")
        assert candidate_to_reranker_text(c) == "Scaled dot-product attention is computed as ..."
        assert candidate_text_source(c) == "chunk_text"

    def test_vlm_caption_only_figure(self) -> None:
        from mrta.retrieval.reranker import candidate_text_source, candidate_to_reranker_text

        c = _figure_cand("figure|d|p4:f1", {"figure_caption": "A diagram of attention."})
        assert candidate_to_reranker_text(c) == "Figure caption: A diagram of attention."
        assert candidate_text_source(c) == "vlm_caption"

    def test_extracted_caption_figure(self) -> None:
        from mrta.retrieval.reranker import candidate_text_source, candidate_to_reranker_text

        c = _figure_cand("figure|d|p4:f1", {"figure_extracted_caption": "Figure 2: Attention."})
        assert candidate_to_reranker_text(c) == "Figure caption: Figure 2: Attention."
        assert candidate_text_source(c) == "extracted_caption"

    def test_nearby_text_fallback_figure(self) -> None:
        from mrta.retrieval.reranker import candidate_text_source, candidate_to_reranker_text

        c = _figure_cand("figure|d|p14:f0", {"figure_nearby_text": "The Law will never be"})
        assert candidate_to_reranker_text(c) == "Context: The Law will never be"
        assert candidate_text_source(c) == "nearby_text_fallback"

    def test_combined_production_text_is_labelled_and_ordered(self) -> None:
        from mrta.retrieval.reranker import candidate_text_source, candidate_to_reranker_text

        c = _figure_cand(
            "figure|d|p4:f1",
            {
                "figure_caption": "A diagram.",
                "figure_description": "Two panels side by side.",
                "figure_nearby_text": "Figure 2: Attention mechanisms.",
            },
        )
        assert candidate_to_reranker_text(c) == (
            "Figure caption: A diagram.\n"
            "Description: Two panels side by side.\n"
            "Context: Figure 2: Attention mechanisms."
        )
        assert candidate_text_source(c) == "combined_production_text"

    def test_vlm_caption_preferred_over_extracted(self) -> None:
        from mrta.retrieval.reranker import candidate_to_reranker_text

        c = _figure_cand(
            "figure|d|p4:f1",
            {"figure_caption": "VLM text.", "figure_extracted_caption": "Extracted text."},
        )
        assert candidate_to_reranker_text(c) == "Figure caption: VLM text."

    def test_empty_figure_payload_is_deterministic_not_an_error(self) -> None:
        from mrta.retrieval.reranker import candidate_text_source, candidate_to_reranker_text

        c = _figure_cand("figure|d|p9:f0", {})
        assert candidate_to_reranker_text(c) == ""
        assert candidate_text_source(c) == "empty"

    def test_whitespace_only_fields_treated_as_empty(self) -> None:
        from mrta.retrieval.reranker import candidate_text_source, candidate_to_reranker_text

        c = _figure_cand("figure|d|p9:f0", {"figure_caption": "   \n  ", "figure_nearby_text": ""})
        assert candidate_to_reranker_text(c) == ""
        assert candidate_text_source(c) == "empty"

    def test_none_valued_payload_fields_handled(self) -> None:
        from mrta.retrieval.reranker import candidate_to_reranker_text

        c = _figure_cand(
            "figure|d|p4:f1", {"figure_caption": None, "figure_nearby_text": "Nearby."}
        )
        assert candidate_to_reranker_text(c) == "Context: Nearby."

    def test_ground_truth_fields_never_reach_model_input(self) -> None:
        """Benchmark annotations in the payload must not leak into model input."""
        from mrta.retrieval.reranker import candidate_to_reranker_text

        c = _figure_cand(
            "figure|d|p4:f1",
            {
                "figure_caption": "A diagram.",
                "source_note": "LEAK-source-note",
                "expected_evidence": "LEAK-expected-evidence",
                "retrieval_challenge": "LEAK-challenge",
                "difficulty": "LEAK-difficulty",
                "figure_id": "LEAK-fig-attention-mechanisms",
            },
        )
        text = candidate_to_reranker_text(c)
        assert "LEAK" not in text
        assert text == "Figure caption: A diagram."

    def test_representation_is_deterministic_across_calls(self) -> None:
        from mrta.retrieval.reranker import candidate_to_reranker_text

        c = _figure_cand("figure|d|p4:f1", {"figure_caption": "A.", "figure_nearby_text": "B."})
        assert len({candidate_to_reranker_text(c) for _ in range(5)}) == 1


class TestCrossEncoderReranker:
    """Ranking behaviour, truncation and determinism (spec 9, 11, 12)."""

    def test_empty_candidates_returns_empty(self) -> None:
        reranker, mock_model = _make_ce_reranker([])
        assert reranker.rerank("q", [], top_k=5) == []
        mock_model.predict.assert_not_called()

    def test_single_candidate(self) -> None:
        reranker, _ = _make_ce_reranker([0.7])
        out = reranker.rerank("q", [_text_cand("text|d|c1", "one")], top_k=5)
        assert len(out) == 1
        assert out[0].reranker_rank == 1
        assert out[0].reranker_score == pytest.approx(0.7)

    def test_multiple_candidates_sorted_descending(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"chunk {i}") for i in range(3)]
        reranker, _ = _make_ce_reranker([0.2, 0.9, 0.5])
        out = reranker.rerank("q", cands, top_k=3)
        assert [r.candidate.canonical_id for r in out] == ["text|d|c1", "text|d|c2", "text|d|c0"]
        assert [r.reranker_rank for r in out] == [1, 2, 3]

    def test_ties_broken_by_rrf_rank_then_canonical_id(self) -> None:
        """Equal scores must fall back to RRF rank, then canonical_id."""
        cands = [
            _text_cand("text|d|c_zebra", "z"),
            _text_cand("text|d|c_alpha", "a"),
            _text_cand("text|d|c_beta", "b"),
        ]
        reranker, _ = _make_ce_reranker([0.5, 0.5, 0.5])
        out = reranker.rerank("q", cands, top_k=3)
        # all tied -> incoming RRF rank (list position) decides
        assert [r.candidate.canonical_id for r in out] == [
            "text|d|c_zebra",
            "text|d|c_alpha",
            "text|d|c_beta",
        ]
        assert [r.original_rrf_rank for r in out] == [1, 2, 3]

    def test_top_k_truncation(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(6)]
        reranker, _ = _make_ce_reranker([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        out = reranker.rerank("q", cands, top_k=2)
        assert len(out) == 2
        assert [r.candidate.canonical_id for r in out] == ["text|d|c5", "text|d|c4"]

    def test_top_k_exceeding_candidate_count(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(2)]
        reranker, _ = _make_ce_reranker([0.4, 0.8])
        out = reranker.rerank("q", cands, top_k=10)
        assert len(out) == 2

    def test_mixed_text_and_figure_pool(self) -> None:
        cands = [
            _text_cand("text|d|c1", "Transformer architecture text."),
            _figure_cand("figure|d|p4:f1", {"figure_caption": "Attention diagram."}),
            _figure_cand("figure|d|p14:f0", {"figure_nearby_text": "boilerplate"}, page=14),
        ]
        reranker, mock_model = _make_ce_reranker([0.1, 0.9, 0.3])
        out = reranker.rerank("q", cands, top_k=3)
        assert [r.candidate.evidence_type for r in out] == ["figure", "figure", "text"]
        assert [r.reranker_text_source for r in out] == [
            "vlm_caption",
            "nearby_text_fallback",
            "chunk_text",
        ]

    def test_repeated_runs_are_deterministic(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(4)]
        results = []
        for _ in range(3):
            reranker, _ = _make_ce_reranker([0.5, 0.5, 0.2, 0.5])
            out = reranker.rerank("q", cands, top_k=4)
            results.append([r.candidate.canonical_id for r in out])
        assert results[0] == results[1] == results[2]


class TestScoreProvenance:
    """RRF and cross-encoder scores must stay separate (spec 10)."""

    def test_rrf_score_preserved_on_result(self) -> None:
        c = _text_cand("text|d|c1", "x", rrf_score=0.0327)
        reranker, _ = _make_ce_reranker([9.5])
        out = reranker.rerank("q", [c], top_k=1)
        assert out[0].original_rrf_score == pytest.approx(0.0327)

    def test_rrf_rank_preserved_on_result(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(3)]
        reranker, _ = _make_ce_reranker([0.1, 0.2, 0.9])
        out = reranker.rerank("q", cands, top_k=3)
        assert out[0].original_rrf_rank == 3  # was last under RRF, first after reranking
        assert out[0].reranker_rank == 1

    def test_reranker_score_stored_separately_from_rrf_score(self) -> None:
        c = _text_cand("text|d|c1", "x", rrf_score=0.0327)
        reranker, _ = _make_ce_reranker([9.5])
        out = reranker.rerank("q", [c], top_k=1)
        assert out[0].reranker_score == pytest.approx(9.5)
        assert out[0].original_rrf_score == pytest.approx(0.0327)
        assert out[0].reranker_score != out[0].original_rrf_score

    def test_underlying_candidate_score_not_overwritten(self) -> None:
        c = _text_cand("text|d|c1", "x", rrf_score=0.0327)
        reranker, _ = _make_ce_reranker([9.5])
        out = reranker.rerank("q", [c], top_k=1)
        assert out[0].candidate.score == pytest.approx(0.0327)

    def test_reranker_rank_is_one_indexed_and_contiguous(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(4)]
        reranker, _ = _make_ce_reranker([0.4, 0.3, 0.2, 0.1])
        out = reranker.rerank("q", cands, top_k=4)
        assert [r.reranker_rank for r in out] == [1, 2, 3, 4]

    def test_canonical_id_preserved(self) -> None:
        c = _figure_cand("figure|attention_is_all_you_need|p4:fig_attention_mechanisms", {})
        reranker, _ = _make_ce_reranker([0.5])
        out = reranker.rerank("q", [c], top_k=1)
        assert (
            out[0].candidate.canonical_id
            == "figure|attention_is_all_you_need|p4:fig_attention_mechanisms"
        )

    def test_modality_provenance_preserved(self) -> None:
        c = _figure_cand(
            "figure|d|p4:f1",
            {"figure_caption": "A."},
            modality_sources=("caption", "clip"),
            source_ranks={"caption": 2, "clip": 7},
        )
        reranker, _ = _make_ce_reranker([0.5])
        out = reranker.rerank("q", [c], top_k=1)
        assert out[0].candidate.modality_sources == ("caption", "clip")
        assert out[0].candidate.source_ranks == {"caption": 2, "clip": 7}

    def test_input_candidates_not_mutated(self) -> None:
        cands = [
            _text_cand("text|d|c1", "one", rrf_score=0.03),
            _figure_cand("figure|d|p4:f1", {"figure_caption": "A."}, rrf_score=0.02),
        ]
        before = [(c.canonical_id, c.score, dict(c.payload)) for c in cands]
        reranker, _ = _make_ce_reranker([0.9, 0.1])
        reranker.rerank("q", cands, top_k=2)
        after = [(c.canonical_id, c.score, dict(c.payload)) for c in cands]
        assert before == after

    def test_input_list_order_not_mutated(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(3)]
        original_order = [c.canonical_id for c in cands]
        reranker, _ = _make_ce_reranker([0.1, 0.9, 0.5])
        reranker.rerank("q", cands, top_k=3)
        assert [c.canonical_id for c in cands] == original_order


class TestBatchScoring:
    """Pair construction and batching (spec 11)."""

    def test_pairs_are_query_text_tuples_in_input_order(self) -> None:
        cands = [
            _text_cand("text|d|c1", "chunk one"),
            _figure_cand("figure|d|p4:f1", {"figure_caption": "A diagram."}),
        ]
        reranker, mock_model = _make_ce_reranker([0.5, 0.6])
        reranker.rerank("what is attention?", cands, top_k=2)
        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [
            ("what is attention?", "chunk one"),
            ("what is attention?", "Figure caption: A diagram."),
        ]

    def test_model_called_once_for_the_whole_batch(self) -> None:
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(8)]
        reranker, mock_model = _make_ce_reranker([0.1] * 8)
        reranker.rerank("q", cands, top_k=5)
        assert mock_model.predict.call_count == 1

    def test_empty_text_still_scored_not_skipped(self) -> None:
        """A textless figure must still be sent to the model, as an empty string."""
        cands = [_text_cand("text|d|c1", "one"), _figure_cand("figure|d|p9:f0", {})]
        reranker, mock_model = _make_ce_reranker([0.2, 0.4])
        out = reranker.rerank("q", cands, top_k=2)
        pairs = mock_model.predict.call_args[0][0]
        assert pairs[1] == ("q", "")
        assert len(out) == 2
        assert out[0].candidate.canonical_id == "figure|d|p9:f0"

    def test_numpy_scores_converted_to_float(self) -> None:
        np = pytest.importorskip("numpy")
        cands = [_text_cand(f"text|d|c{i}", f"c{i}") for i in range(2)]
        reranker, _ = _make_ce_reranker(np.array([0.25, 0.75], dtype="float32"))
        out = reranker.rerank("q", cands, top_k=2)
        assert isinstance(out[0].reranker_score, float)
        assert out[0].reranker_score == pytest.approx(0.75)

    def test_model_name_pinned_from_settings_by_default(self) -> None:
        from mrta.core.config import settings
        from mrta.retrieval.reranker import CrossEncoderReranker

        with patch("sentence_transformers.CrossEncoder") as mock_cls:
            mock_cls.return_value = MagicMock()
            reranker = CrossEncoderReranker()
        assert reranker.model_name == settings.reranker_model_name
        assert settings.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


class TestAdapterRoundTrip:
    """Reranked candidates must still score through the evaluation adapter."""

    def test_adapter_conversion_preserves_canonical_evidence(self) -> None:
        from mrta.eval.adapter import EvalAdapter

        manifest = {
            "documents": [
                {
                    "filename": "attention.pdf",
                    "document_id": "attention_is_all_you_need",
                    "figures": [
                        {
                            "page_number": 4,
                            "figure_index": 1,
                            "figure_id": "fig_attention_mechanisms",
                        }
                    ],
                }
            ]
        }
        adapter = EvalAdapter(manifest)
        c = _figure_cand(
            "figure|attention_is_all_you_need|p4:fig_attention_mechanisms",
            {"figure_caption": "A diagram."},
            page=4,
            rrf_score=0.031,
        )
        reranker, _ = _make_ce_reranker([0.9])
        out = reranker.rerank("q", [c], top_k=1)

        scored = adapter.from_fused_candidate(out[0].candidate, rank=out[0].reranker_rank)
        assert scored.evidence.document_id == "attention_is_all_you_need"
        assert scored.evidence.page_number == 4
        assert scored.evidence.figure_id == "fig_attention_mechanisms"
        assert scored.rank == 1
        # the adapter carries the RRF score through, not the cross-encoder score
        assert scored.score == pytest.approx(0.031)
