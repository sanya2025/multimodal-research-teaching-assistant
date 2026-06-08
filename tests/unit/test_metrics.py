"""Tests for mrta.evaluation.metrics — all metrics return floats in [0,1]."""

from __future__ import annotations

from mrta.core.schemas import Chunk
from mrta.evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    faithfulness,
    hallucination_rate,
)

CHUNKS = [
    Chunk(
        chunk_id="d1_p3_c0",
        doc_id="d1",
        source="paper.pdf",
        page=3,
        text="The model uses 512 dimensions.",
    ),
    Chunk(
        chunk_id="d1_p5_c1",
        doc_id="d1",
        source="paper.pdf",
        page=5,
        text="Multi-head attention with 8 heads.",
    ),
]


class TestAnswerRelevance:
    def test_keyword_present(self) -> None:
        score = answer_relevance("What is 512?", "The dimension is 512.")
        assert score == 1.0

    def test_no_overlap(self) -> None:
        score = answer_relevance("What is 512?", "This says nothing relevant.")
        assert 0.0 <= score <= 1.0

    def test_returns_float(self) -> None:
        assert isinstance(answer_relevance("Q", "A"), float)

    def test_range(self) -> None:
        assert 0.0 <= answer_relevance("dimensions model?", "unrelated text here") <= 1.0


class TestFaithfulness:
    def test_grounded_answer(self) -> None:
        score = faithfulness("The model uses 512 dimensions.", CHUNKS)
        assert score == 1.0

    def test_empty_answer(self) -> None:
        assert faithfulness("", CHUNKS) == 1.0

    def test_returns_float(self) -> None:
        assert isinstance(faithfulness("answer", CHUNKS), float)

    def test_range(self) -> None:
        assert 0.0 <= faithfulness("some text about things", CHUNKS) <= 1.0

    def test_ungrounded_sentence(self) -> None:
        score = faithfulness("Completely invented claim about dragons flying.", CHUNKS)
        assert 0.0 <= score <= 1.0


class TestCitationCorrectness:
    def test_correct_citation(self) -> None:
        assert citation_correctness("The answer is in [page 3].", CHUNKS) == 1.0

    def test_wrong_page(self) -> None:
        assert citation_correctness("See [page 99].", CHUNKS) == 0.0

    def test_no_citations(self) -> None:
        assert citation_correctness("Answer with no citations.", CHUNKS) == 1.0

    def test_mixed_citations(self) -> None:
        score = citation_correctness("See [page 3] and [page 99].", CHUNKS)
        assert score == 0.5

    def test_returns_float(self) -> None:
        assert isinstance(citation_correctness("text", CHUNKS), float)


class TestHallucinationRate:
    def test_complement_of_faithfulness(self) -> None:
        answer = "The model uses 512 dimensions."
        assert abs(faithfulness(answer, CHUNKS) + hallucination_rate(answer, CHUNKS) - 1.0) < 1e-9

    def test_range(self) -> None:
        assert 0.0 <= hallucination_rate("some text", CHUNKS) <= 1.0

    def test_returns_float(self) -> None:
        assert isinstance(hallucination_rate("answer", CHUNKS), float)
