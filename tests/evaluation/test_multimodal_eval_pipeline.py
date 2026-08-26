"""Unit tests for run_multimodal_eval — all MultimodalRAG calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mrta.core.schemas import MultimodalAnswer, MultimodalCitation
from mrta.evaluation.multimodal_eval_pipeline import MultimodalEvalReport, run_multimodal_eval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_answer(
    answer_text: str = "The answer.",
    text_page: int = 1,
    visual_page: int | None = None,
) -> MultimodalAnswer:
    text_cits = [
        MultimodalCitation(
            label="[T1]",
            evidence_id=f"t_{text_page}",
            modality="text",
            source="sample.pdf",
            page=text_page,
        )
    ]
    visual_cits = []
    if visual_page is not None:
        visual_cits = [
            MultimodalCitation(
                label="[V1]",
                evidence_id=f"v_{visual_page}",
                modality="image",
                source="sample.pdf",
                page=visual_page,
                figure_index=1,
            )
        ]
    return MultimodalAnswer(
        answer=answer_text,
        text_citations=text_cits,
        visual_citations=visual_cits,
        retrieval_mode="multimodal",
        latency_s=0.5,
    )


def _mock_rag(answer: MultimodalAnswer) -> MagicMock:
    rag = MagicMock()
    rag.ask.return_value = answer
    return rag


SIMPLE_BENCHMARK = [
    {
        "question": "What is attention?",
        "question_type": "text_only",
        "relevant_text_pages": [1],
        "relevant_figures": [],
    }
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunMultimodalEval:
    def test_returns_multimodal_eval_report(self) -> None:
        rag = _mock_rag(_make_answer())
        result = run_multimodal_eval(SIMPLE_BENCHMARK, rag)
        assert isinstance(result, MultimodalEvalReport)

    def test_n_questions_correct(self) -> None:
        rag = _mock_rag(_make_answer())
        result = run_multimodal_eval(SIMPLE_BENCHMARK * 3, rag)
        assert result.n_questions == 3

    def test_rag_ask_called_once_per_item(self) -> None:
        rag = _mock_rag(_make_answer())
        run_multimodal_eval(SIMPLE_BENCHMARK * 4, rag)
        assert rag.ask.call_count == 4

    def test_latency_populated(self) -> None:
        rag = _mock_rag(_make_answer())
        result = run_multimodal_eval(SIMPLE_BENCHMARK, rag)
        assert result.mean_latency_s == pytest.approx(0.5)

    def test_figure_recall_perfect_when_no_figures_expected(self) -> None:
        rag = _mock_rag(_make_answer())
        result = run_multimodal_eval(SIMPLE_BENCHMARK, rag)
        assert result.figure_recall_at_5 == 1.0

    def test_text_recall_perfect_when_page_retrieved(self) -> None:
        rag = _mock_rag(_make_answer(text_page=1))
        result = run_multimodal_eval(SIMPLE_BENCHMARK, rag)
        assert result.text_recall_at_k == 1.0

    def test_by_type_populated(self) -> None:
        rag = _mock_rag(_make_answer())
        result = run_multimodal_eval(SIMPLE_BENCHMARK, rag)
        assert "text_only" in result.by_type

    def test_by_type_metrics_keys(self) -> None:
        rag = _mock_rag(_make_answer())
        result = run_multimodal_eval(SIMPLE_BENCHMARK, rag)
        row = result.by_type["text_only"]
        assert "figure_recall_at_5" in row
        assert "overall_recall" in row
        assert "answer_relevance" in row

    def test_empty_benchmark_returns_zeros(self) -> None:
        rag = MagicMock()
        result = run_multimodal_eval([], rag)
        assert result.n_questions == 0
        assert result.mean_latency_s == 0.0

    def test_visual_recall_zero_when_figure_not_retrieved(self) -> None:
        bench = [
            {
                "question": "Describe Figure 1.",
                "question_type": "figure_lookup",
                "relevant_text_pages": [],
                "relevant_figures": [{"source": "sample.pdf", "page": 99, "figure_index": 1}],
            }
        ]
        rag = _mock_rag(_make_answer(visual_page=2))  # wrong page
        result = run_multimodal_eval(bench, rag)
        assert result.visual_recall_at_k == 0.0
