"""Unit tests for mrta.eval.retrieval_metrics.

Covers: Recall@k, Hit@k, MRR, nDCG@k with edge cases, partial matches,
deduplication, and bound checks (0 <= metric <= 1).
"""

from __future__ import annotations

import pytest

from mrta.eval.retrieval_metrics import (
    figure_recall_at_k,
    hit_rate_at_k,
    is_hit,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from mrta.eval.types import CanonicalEvidence, RetrievedCandidate

DOC = "doc_attention_2017"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ev(page: int, figure_id: str | None = None, doc: str = DOC) -> CanonicalEvidence:
    return CanonicalEvidence(document_id=doc, page_number=page, figure_id=figure_id)


def cand(
    page: int, figure_id: str | None = None, score: float = 0.9, rank: int = 1, doc: str = DOC
) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=f"{doc}_p{page}_f{figure_id}",
        evidence=ev(page, figure_id, doc),
        score=score,
        rank=rank,
    )


def ranked(
    *pages_or_cands: int | RetrievedCandidate, figure_id: str | None = None
) -> list[RetrievedCandidate]:
    result = []
    for i, item in enumerate(pages_or_cands, start=1):
        if isinstance(item, RetrievedCandidate):
            result.append(item)
        else:
            result.append(cand(item, figure_id=figure_id, score=1.0 - i * 0.05, rank=i))
    return result


# ---------------------------------------------------------------------------
# 1. Perfect retrieval at rank 1
# ---------------------------------------------------------------------------


class TestPerfectRetrieval:
    def test_recall_perfect(self) -> None:
        assert recall_at_k(ranked(3), [ev(3)], k=5) == 1.0

    def test_hit_perfect(self) -> None:
        assert hit_rate_at_k(ranked(3), [ev(3)], k=5) == 1.0

    def test_mrr_perfect(self) -> None:
        assert mean_reciprocal_rank(ranked(3), [ev(3)]) == pytest.approx(1.0)

    def test_ndcg_perfect(self) -> None:
        assert ndcg_at_k(ranked(3), [ev(3)], k=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Relevant result at rank 2
# ---------------------------------------------------------------------------


class TestRelevantAtRank2:
    def test_mrr_rank2(self) -> None:
        assert mean_reciprocal_rank(ranked(99, 3), [ev(3)]) == pytest.approx(0.5)

    def test_ndcg_rank2_less_than_rank1(self) -> None:
        rank1 = ndcg_at_k(ranked(3, 99), [ev(3)], k=5)
        rank2 = ndcg_at_k(ranked(99, 3), [ev(3)], k=5)
        assert rank1 > rank2

    def test_recall_rank2_counts(self) -> None:
        assert recall_at_k(ranked(99, 3), [ev(3)], k=5) == 1.0

    def test_recall_rank2_outside_k(self) -> None:
        assert recall_at_k(ranked(99, 3), [ev(3)], k=1) == 0.0


# ---------------------------------------------------------------------------
# 3. Multiple targets with partial recall
# ---------------------------------------------------------------------------


class TestMultipleTargetsPartialRecall:
    def test_recall_half(self) -> None:
        candidates = ranked(3, 99, 99)
        targets = [ev(3), ev(4)]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(0.5)

    def test_recall_full_two_targets(self) -> None:
        candidates = ranked(3, 4)
        targets = [ev(3), ev(4)]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)

    def test_hit_with_partial_recall(self) -> None:
        candidates = ranked(3, 99)
        targets = [ev(3), ev(4)]
        assert hit_rate_at_k(candidates, targets, k=5) == 1.0

    def test_mrr_first_match_wins(self) -> None:
        candidates = ranked(99, 3, 4)
        targets = [ev(3), ev(4)]
        assert mean_reciprocal_rank(candidates, targets) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. Empty candidate list
# ---------------------------------------------------------------------------


class TestEmptyCandidates:
    def test_recall_empty_candidates(self) -> None:
        assert recall_at_k([], [ev(3)], k=5) == 0.0

    def test_hit_empty_candidates(self) -> None:
        assert hit_rate_at_k([], [ev(3)], k=5) == 0.0

    def test_mrr_empty_candidates(self) -> None:
        assert mean_reciprocal_rank([], [ev(3)]) == 0.0

    def test_ndcg_empty_candidates(self) -> None:
        assert ndcg_at_k([], [ev(3)], k=5) == 0.0


# ---------------------------------------------------------------------------
# 5. Irrelevant candidates only
# ---------------------------------------------------------------------------


class TestIrrelevantCandidates:
    def test_recall_zero(self) -> None:
        assert recall_at_k(ranked(99, 98, 97), [ev(3)], k=5) == 0.0

    def test_hit_zero(self) -> None:
        assert hit_rate_at_k(ranked(99, 98), [ev(3)], k=5) == 0.0

    def test_mrr_zero(self) -> None:
        assert mean_reciprocal_rank(ranked(99), [ev(3)]) == 0.0

    def test_ndcg_zero(self) -> None:
        assert ndcg_at_k(ranked(99), [ev(3)], k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Correct text page matching
# ---------------------------------------------------------------------------


def test_text_page_match() -> None:
    target = ev(2)
    candidate = cand(2)
    assert target.matches(candidate.evidence)


def test_text_page_mismatch() -> None:
    target = ev(2)
    candidate = cand(3)
    assert not target.matches(candidate.evidence)


def test_text_doc_mismatch() -> None:
    target = ev(2, doc="doc_a")
    candidate = cand(2, doc="doc_b")
    assert not target.matches(candidate.evidence)


# ---------------------------------------------------------------------------
# 7. Correct figure-level matching
# ---------------------------------------------------------------------------


def test_figure_match_exact() -> None:
    target = ev(3, "fig_transformer_arch")
    candidate = cand(3, "fig_transformer_arch")
    assert target.matches(candidate.evidence)


def test_figure_match_different_figure() -> None:
    target = ev(3, "fig_transformer_arch")
    candidate = cand(3, "fig_attention_mechanisms")
    assert not target.matches(candidate.evidence)


# ---------------------------------------------------------------------------
# 8. Wrong figure on correct page
# ---------------------------------------------------------------------------


def test_wrong_figure_on_correct_page() -> None:
    target = ev(3, "fig_transformer_arch")
    candidate_wrong_fig = cand(3, "fig_other")
    candidate_no_fig = cand(3, None)
    assert not target.matches(candidate_wrong_fig.evidence)
    assert not target.matches(candidate_no_fig.evidence)


# ---------------------------------------------------------------------------
# 9. Wrong page
# ---------------------------------------------------------------------------


def test_wrong_page_no_match() -> None:
    target = ev(3)
    candidate = cand(4)
    assert not target.matches(candidate.evidence)


# ---------------------------------------------------------------------------
# 10. Wrong document
# ---------------------------------------------------------------------------


def test_wrong_document_no_match() -> None:
    target = ev(3, doc="doc_attention_2017")
    candidate = cand(3, doc="doc_other")
    assert not target.matches(candidate.evidence)


# ---------------------------------------------------------------------------
# 11. k cutoff behavior
# ---------------------------------------------------------------------------


class TestKCutoff:
    def test_recall_respects_k(self) -> None:
        candidates = ranked(99, 99, 99, 3)
        assert recall_at_k(candidates, [ev(3)], k=3) == 0.0
        assert recall_at_k(candidates, [ev(3)], k=4) == 1.0

    def test_ndcg_respects_k(self) -> None:
        candidates = ranked(99, 99, 99, 3)
        assert ndcg_at_k(candidates, [ev(3)], k=3) == pytest.approx(0.0)
        assert ndcg_at_k(candidates, [ev(3)], k=4) > 0.0

    def test_is_hit_respects_k(self) -> None:
        candidates = ranked(99, 3)
        assert not is_hit(candidates, [ev(3)], k=1)
        assert is_hit(candidates, [ev(3)], k=2)


# ---------------------------------------------------------------------------
# 12. nDCG normalization — cannot exceed 1.0
# ---------------------------------------------------------------------------


def test_ndcg_never_exceeds_one_single_target() -> None:
    candidates = ranked(3, 3, 3, 3, 3)
    for c in candidates:
        c.evidence = ev(3)
    assert ndcg_at_k(candidates, [ev(3)], k=5) <= 1.0


def test_ndcg_never_exceeds_one_two_targets() -> None:
    candidates = [cand(3, rank=i + 1, score=0.9 - i * 0.05) for i in range(5)]
    targets = [ev(3), ev(4)]
    result = ndcg_at_k(candidates, targets, k=5)
    assert result <= 1.0


# ---------------------------------------------------------------------------
# 13. Duplicate retrieved chunks mapping to same canonical evidence
# ---------------------------------------------------------------------------


class TestDuplicateCandidates:
    def test_duplicates_do_not_inflate_recall(self) -> None:
        # Five chunks all on page 3 — should count as 1 hit, not 5
        candidates = [cand(3, score=0.9 - i * 0.05, rank=i + 1) for i in range(5)]
        targets = [ev(3)]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)

    def test_duplicates_do_not_inflate_ndcg_above_one(self) -> None:
        candidates = [cand(3, score=0.9 - i * 0.05, rank=i + 1) for i in range(5)]
        targets = [ev(3)]
        result = ndcg_at_k(candidates, targets, k=5)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(1.0)

    def test_duplicate_page_one_target_not_overcounted(self) -> None:
        # 3 candidates on page 3, 2 on page 4; only 1 target (page 3)
        candidates = [
            cand(3, rank=1, score=0.9),
            cand(3, rank=2, score=0.8),
            cand(4, rank=3, score=0.7),
        ]
        targets = [ev(3)]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)
        assert ndcg_at_k(candidates, targets, k=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 14. Multiple ground-truth evidence units
# ---------------------------------------------------------------------------


class TestMultipleTargets:
    def test_two_targets_both_found(self) -> None:
        candidates = [cand(3, rank=1), cand(4, rank=2)]
        targets = [ev(3), ev(4)]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)

    def test_two_targets_one_found(self) -> None:
        candidates = [cand(3, rank=1), cand(99, rank=2)]
        targets = [ev(3), ev(4)]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(0.5)

    def test_hybrid_figure_and_text_both_match(self) -> None:
        candidates = [
            cand(3, "fig_transformer_arch", rank=1),
            cand(3, None, rank=2),
        ]
        targets = [
            ev(3, "fig_transformer_arch"),
            ev(3, None),
        ]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)

    def test_hybrid_only_text_matched(self) -> None:
        candidates = [cand(3, None, rank=1)]
        targets = [ev(3, "fig_transformer_arch"), ev(3, None)]
        assert recall_at_k(candidates, targets, k=5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 15. Empty target behavior
# ---------------------------------------------------------------------------


class TestEmptyTargets:
    def test_recall_empty_targets(self) -> None:
        assert recall_at_k(ranked(3), [], k=5) == 1.0

    def test_hit_empty_targets(self) -> None:
        assert hit_rate_at_k(ranked(3), [], k=5) == 1.0

    def test_mrr_empty_targets(self) -> None:
        assert mean_reciprocal_rank(ranked(3), []) == 0.0

    def test_ndcg_empty_targets(self) -> None:
        assert ndcg_at_k(ranked(3), [], k=5) == 1.0


# ---------------------------------------------------------------------------
# figure_recall_at_k
# ---------------------------------------------------------------------------


class TestFigureRecallAtK:
    def test_figure_target_found(self) -> None:
        candidates = ranked(3, figure_id="fig_transformer_arch")
        targets = [ev(3, "fig_transformer_arch")]
        assert figure_recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)

    def test_figure_target_not_found(self) -> None:
        candidates = ranked(3)  # text chunk, figure_id=None
        targets = [ev(3, "fig_transformer_arch")]
        assert figure_recall_at_k(candidates, targets, k=5) == pytest.approx(0.0)

    def test_no_figure_targets_returns_one(self) -> None:
        # All targets are text (figure_id=None): nothing figure-specific to recall.
        # The metric returns 1.0 — callers should treat this as N/A in reporting
        # rather than inferring perfect figure retrieval.
        candidates = ranked(3, 4)
        targets = [ev(3), ev(4)]
        assert figure_recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)

    def test_mixed_targets_only_counts_figure_targets(self) -> None:
        # One figure target + one text target; only the figure target is evaluated.
        fig_cand = cand(3, "fig_transformer_arch", rank=1)
        text_cand = cand(3, None, rank=2)
        candidates = [fig_cand, text_cand]
        targets = [ev(3, "fig_transformer_arch"), ev(3)]
        # figure_recall_at_k filters to [ev(3, "fig_transformer_arch")] → found at rank 1
        assert figure_recall_at_k(candidates, targets, k=5) == pytest.approx(1.0)

    def test_mixed_targets_figure_missing(self) -> None:
        text_cand = cand(3, None, rank=1)
        candidates = [text_cand]
        targets = [ev(3, "fig_transformer_arch"), ev(3)]
        # figure target not retrieved → figure_recall = 0
        assert figure_recall_at_k(candidates, targets, k=5) == pytest.approx(0.0)

    def test_k_cutoff_respected(self) -> None:
        candidates = ranked(99, 99, 3, figure_id="fig_transformer_arch")
        targets = [ev(3, "fig_transformer_arch")]
        assert figure_recall_at_k(candidates, targets, k=2) == pytest.approx(0.0)
        assert figure_recall_at_k(candidates, targets, k=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bound checks — all metrics in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 5])
def test_all_metrics_bounded(k: int) -> None:
    candidates = ranked(3, 4, 99, 98, 97)
    targets = [ev(3), ev(4)]
    r = recall_at_k(candidates, targets, k=k)
    h = hit_rate_at_k(candidates, targets, k=k)
    m = mean_reciprocal_rank(candidates, targets)
    n = ndcg_at_k(candidates, targets, k=k)
    assert 0.0 <= r <= 1.0
    assert 0.0 <= h <= 1.0
    assert 0.0 <= m <= 1.0
    assert 0.0 <= n <= 1.0
