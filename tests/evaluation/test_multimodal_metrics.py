"""Unit tests for multimodal_metrics — figure_recall_at_k, multimodal_recall_at_k,
multimodal_citation_correctness.

All tests use synthetic EvidenceRecord and MultimodalAnswer fixtures.
No Ollama, CLIP, or FAISS required.
"""

from __future__ import annotations

import math

import pytest

from mrta.core.schemas import EvidenceRecord, MultimodalAnswer, MultimodalCitation
from mrta.evaluation.multimodal_metrics import (
    figure_recall_at_k,
    multimodal_citation_correctness,
    multimodal_recall_at_k,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _text_record(page: int, source: str = "paper.pdf") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"{source}_p{page}_text",
        doc_id="doc1",
        source=source,
        page=page,
        modality="text",
        text="Some text.",
    )


def _image_record(
    page: int, figure_index: int | None = 1, source: str = "paper.pdf"
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"{source}_p{page}_f{figure_index}",
        doc_id="doc1",
        source=source,
        page=page,
        modality="image",
        figure_index=figure_index,
        image_bytes=b"fake",
    )


def _answer(
    text_citations: list[MultimodalCitation] | None = None,
    visual_citations: list[MultimodalCitation] | None = None,
    answer_text: str = "",
) -> MultimodalAnswer:
    return MultimodalAnswer(
        answer=answer_text,
        text_citations=text_citations or [],
        visual_citations=visual_citations or [],
        retrieval_mode="multimodal",
        latency_s=0.1,
    )


# ---------------------------------------------------------------------------
# figure_recall_at_k
# ---------------------------------------------------------------------------


class TestFigureRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = [_image_record(2, figure_index=1), _image_record(3, figure_index=2)]
        expected = [{"source": "paper.pdf", "page": 2, "figure_index": 1}]
        assert figure_recall_at_k(retrieved, expected, k=5) == 1.0

    def test_zero_recall_no_match(self) -> None:
        retrieved = [_image_record(2, figure_index=1)]
        expected = [{"source": "paper.pdf", "page": 5, "figure_index": 3}]
        assert figure_recall_at_k(retrieved, expected, k=5) == 0.0

    def test_partial_recall(self) -> None:
        retrieved = [_image_record(2, figure_index=1)]
        expected = [
            {"source": "paper.pdf", "page": 2, "figure_index": 1},
            {"source": "paper.pdf", "page": 3, "figure_index": 2},
        ]
        assert figure_recall_at_k(retrieved, expected, k=5) == pytest.approx(0.5)

    def test_k_cutoff_respected(self) -> None:
        # figure at rank 4 should not count at k=3
        retrieved = [
            _text_record(1),
            _text_record(2),
            _text_record(3),
            _image_record(2, figure_index=1),
        ]
        expected = [{"source": "paper.pdf", "page": 2, "figure_index": 1}]
        assert figure_recall_at_k(retrieved, expected, k=3) == 0.0

    def test_empty_expected_returns_one(self) -> None:
        retrieved = [_image_record(2)]
        assert figure_recall_at_k(retrieved, [], k=5) == 1.0

    def test_no_visual_in_retrieved(self) -> None:
        retrieved = [_text_record(1), _text_record(2)]
        expected = [{"source": "paper.pdf", "page": 2, "figure_index": 1}]
        assert figure_recall_at_k(retrieved, expected, k=5) == 0.0

    def test_source_must_match(self) -> None:
        retrieved = [_image_record(2, figure_index=1, source="other.pdf")]
        expected = [{"source": "paper.pdf", "page": 2, "figure_index": 1}]
        assert figure_recall_at_k(retrieved, expected, k=5) == 0.0

    def test_none_figure_index_matches(self) -> None:
        retrieved = [_image_record(2, figure_index=None)]
        expected = [{"source": "paper.pdf", "page": 2, "figure_index": None}]
        assert figure_recall_at_k(retrieved, expected, k=5) == 1.0


# ---------------------------------------------------------------------------
# multimodal_recall_at_k
# ---------------------------------------------------------------------------


class TestMultimodalRecallAtK:
    def test_perfect_text_and_visual(self) -> None:
        retrieved = [_text_record(1), _image_record(2, figure_index=1)]
        result = multimodal_recall_at_k(
            retrieved,
            expected_text_pages=[1],
            expected_figures=[{"source": "paper.pdf", "page": 2, "figure_index": 1}],
            k=5,
        )
        assert result["text"] == 1.0
        assert result["visual"] == 1.0
        assert result["overall"] == pytest.approx(1.0)

    def test_zero_visual_drags_overall_to_zero(self) -> None:
        retrieved = [_text_record(1)]
        result = multimodal_recall_at_k(
            retrieved,
            expected_text_pages=[1],
            expected_figures=[{"source": "paper.pdf", "page": 2, "figure_index": 1}],
            k=5,
        )
        assert result["text"] == 1.0
        assert result["visual"] == 0.0
        assert result["overall"] == 0.0

    def test_overall_is_geometric_mean(self) -> None:
        retrieved = [_text_record(1), _image_record(2, figure_index=1)]
        result = multimodal_recall_at_k(
            retrieved,
            expected_text_pages=[1, 3],
            expected_figures=[{"source": "paper.pdf", "page": 2, "figure_index": 1}],
            k=5,
        )
        assert result["text"] == pytest.approx(0.5)
        assert result["visual"] == pytest.approx(1.0)
        assert result["overall"] == pytest.approx(math.sqrt(0.5 * 1.0))

    def test_empty_expected_text_returns_one(self) -> None:
        retrieved = [_text_record(1)]
        result = multimodal_recall_at_k(retrieved, [], [], k=5)
        assert result["text"] == 1.0
        assert result["visual"] == 1.0

    def test_source_filter_applied(self) -> None:
        retrieved = [_text_record(1, source="other.pdf"), _text_record(1, source="paper.pdf")]
        result = multimodal_recall_at_k(
            retrieved, expected_text_pages=[1], expected_figures=[], k=5, source="paper.pdf"
        )
        assert result["text"] == 1.0

    def test_keys_present(self) -> None:
        result = multimodal_recall_at_k([], [], [], k=5)
        assert set(result.keys()) == {"text", "visual", "overall"}


# ---------------------------------------------------------------------------
# multimodal_citation_correctness
# ---------------------------------------------------------------------------


class TestMultimodalCitationCorrectness:
    def _make_text_citation(self, label: str, page: int) -> MultimodalCitation:
        return MultimodalCitation(
            label=label,
            evidence_id=f"ev_{page}",
            modality="text",
            source="paper.pdf",
            page=page,
        )

    def _make_visual_citation(self, label: str, page: int) -> MultimodalCitation:
        return MultimodalCitation(
            label=label,
            evidence_id=f"ev_v{page}",
            modality="image",
            source="paper.pdf",
            page=page,
            figure_index=1,
        )

    def test_perfect_citations(self) -> None:
        ans = _answer(
            text_citations=[self._make_text_citation("[T1]", 1)],
            visual_citations=[self._make_visual_citation("[V1]", 2)],
            answer_text="See [T1] and [V1] for details.",
        )
        retrieved = [_text_record(1), _image_record(2)]
        result = multimodal_citation_correctness(ans, retrieved)
        assert result["format"] == 1.0
        assert result["provenance"] == 1.0
        assert result["support"] == 1.0
        assert result["overall"] == pytest.approx(1.0)

    def test_no_citations_in_answer(self) -> None:
        ans = _answer(answer_text="No citations here.")
        retrieved = [_text_record(1)]
        result = multimodal_citation_correctness(ans, retrieved)
        assert result["format"] == 1.0
        assert result["provenance"] == 1.0
        assert result["support"] == 1.0

    def test_cited_page_not_in_retrieved(self) -> None:
        ans = _answer(
            text_citations=[self._make_text_citation("[T1]", 99)],
            answer_text="See [T1].",
        )
        retrieved = [_text_record(1)]
        result = multimodal_citation_correctness(ans, retrieved)
        assert result["support"] < 1.0

    def test_keys_present(self) -> None:
        ans = _answer(answer_text="")
        result = multimodal_citation_correctness(ans, [])
        assert set(result.keys()) == {"format", "provenance", "support", "overall"}

    def test_overall_is_mean_of_three(self) -> None:
        ans = _answer(
            text_citations=[self._make_text_citation("[T1]", 1)],
            answer_text="See [T1].",
        )
        retrieved = [_text_record(1)]
        result = multimodal_citation_correctness(ans, retrieved)
        expected_overall = (result["format"] + result["provenance"] + result["support"]) / 3
        assert result["overall"] == pytest.approx(expected_overall)
