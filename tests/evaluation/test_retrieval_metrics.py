"""Pure unit tests for recall_at_k, mrr, ndcg_at_k, and citation_coverage.

No golden QA, no YAML, no Chunk objects — just plain list[str] inputs.
"""

from __future__ import annotations

import pytest

from mrta.evaluation.metrics import citation_coverage, mrr, ndcg_at_k, recall_at_k

# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_perfect() -> None:
    retrieved = ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]
    expected = ["a.pdf", "b.pdf"]
    assert recall_at_k(retrieved, expected, k=5) == 1.0


def test_recall_at_k_partial() -> None:
    retrieved = ["a.pdf", "x.pdf", "y.pdf"]
    expected = ["a.pdf", "z.pdf"]
    assert recall_at_k(retrieved, expected, k=3) == pytest.approx(0.5)


def test_recall_at_k_none() -> None:
    retrieved = ["x.pdf", "y.pdf"]
    expected = ["a.pdf"]
    assert recall_at_k(retrieved, expected, k=5) == 0.0


def test_recall_at_k_respects_cutoff() -> None:
    # expected doc is at rank 4 but k=3, so it must not be counted
    retrieved = ["x.pdf", "y.pdf", "z.pdf", "a.pdf"]
    expected = ["a.pdf"]
    assert recall_at_k(retrieved, expected, k=3) == 0.0


def test_recall_at_k_empty_expected() -> None:
    assert recall_at_k(["a.pdf"], [], k=5) == 1.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


def test_mrr_first_rank() -> None:
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    expected = ["a.pdf"]
    assert mrr(retrieved, expected) == pytest.approx(1.0)


def test_mrr_second_rank() -> None:
    retrieved = ["x.pdf", "a.pdf", "b.pdf"]
    expected = ["a.pdf"]
    assert mrr(retrieved, expected) == pytest.approx(0.5)


def test_mrr_no_match() -> None:
    retrieved = ["x.pdf", "y.pdf"]
    expected = ["a.pdf"]
    assert mrr(retrieved, expected) == 0.0


# ---------------------------------------------------------------------------
# citation_coverage
# ---------------------------------------------------------------------------


def test_citation_coverage_full() -> None:
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    expected = ["a.pdf", "b.pdf"]
    assert citation_coverage(retrieved, expected) == 1.0


def test_citation_coverage_partial() -> None:
    retrieved = ["a.pdf", "x.pdf"]
    expected = ["a.pdf", "b.pdf"]
    assert citation_coverage(retrieved, expected) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_at_k_rank1_beats_rank3() -> None:
    expected = ["a.pdf"]
    score_rank1 = ndcg_at_k(["a.pdf", "x.pdf", "y.pdf"], expected, k=5)
    score_rank3 = ndcg_at_k(["x.pdf", "y.pdf", "a.pdf"], expected, k=5)
    assert score_rank1 > score_rank3


def test_ndcg_at_k_empty_expected() -> None:
    assert ndcg_at_k(["a.pdf", "b.pdf"], [], k=5) == 1.0


def test_ndcg_at_k_perfect() -> None:
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    expected = ["a.pdf", "b.pdf"]
    assert ndcg_at_k(retrieved, expected, k=5) == pytest.approx(1.0)


def test_ndcg_at_k_no_match() -> None:
    retrieved = ["x.pdf", "y.pdf"]
    expected = ["a.pdf"]
    assert ndcg_at_k(retrieved, expected, k=5) == pytest.approx(0.0)
