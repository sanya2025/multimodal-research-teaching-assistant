"""Unit tests for reciprocal_rank_fusion and FusedResult.

No model dependencies — fusion is a pure function over EvidenceRecord lists.
"""

from __future__ import annotations

import pytest

from mrta.core.schemas import EvidenceRecord
from mrta.retrieval.fusion import FusedResult, reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(eid: str, modality: str = "text") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        doc_id="doc1",
        source="test.pdf",
        page=1,
        modality=modality,  # type: ignore[arg-type]
        text="some text" if modality == "text" else None,
    )


# ---------------------------------------------------------------------------
# TestFusedResult
# ---------------------------------------------------------------------------


class TestFusedResult:
    def test_dataclass_fields(self) -> None:
        rec = make_record("r1")
        fr = FusedResult(
            record=rec, rrf_score=0.5, per_list_rank={"text": 1}, source_modality="text"
        )
        assert fr.record is rec
        assert fr.rrf_score == pytest.approx(0.5)
        assert fr.per_list_rank == {"text": 1}
        assert fr.source_modality == "text"

    def test_defaults(self) -> None:
        rec = make_record("r1")
        fr = FusedResult(record=rec, rrf_score=0.0)
        assert fr.per_list_rank == {}
        assert fr.source_modality == "text"


# ---------------------------------------------------------------------------
# TestRRF
# ---------------------------------------------------------------------------


class TestRRF:
    def test_empty_named_lists_returns_empty(self) -> None:
        assert reciprocal_rank_fusion({}) == []

    def test_all_empty_lists_returns_empty(self) -> None:
        assert reciprocal_rank_fusion({"text": [], "visual": []}) == []

    def test_single_list_passthrough(self) -> None:
        r1, r2 = make_record("r1"), make_record("r2")
        results = reciprocal_rank_fusion({"text": [r1, r2]})
        assert len(results) == 2
        assert results[0].record.evidence_id == "r1"
        assert results[1].record.evidence_id == "r2"

    def test_rrf_score_formula_single_list(self) -> None:
        r1 = make_record("r1")
        results = reciprocal_rank_fusion({"text": [r1]}, k=60)
        assert results[0].rrf_score == pytest.approx(1.0 / (60 + 1))

    def test_rrf_score_formula_two_lists(self) -> None:
        r1 = make_record("r1")
        results = reciprocal_rank_fusion({"text": [r1], "visual": [r1]}, k=60)
        expected = 1.0 / 61 + 1.0 / 61
        assert results[0].rrf_score == pytest.approx(expected)

    def test_deduplication_by_evidence_id(self) -> None:
        r1 = make_record("r1")
        results = reciprocal_rank_fusion({"text": [r1], "visual": [r1]})
        assert len(results) == 1
        assert results[0].record.evidence_id == "r1"

    def test_shared_doc_ranks_higher_than_exclusive_docs(self) -> None:
        shared = make_record("shared")
        text_only = make_record("text_only")
        visual_only = make_record("visual_only")
        results = reciprocal_rank_fusion(
            {"text": [shared, text_only], "visual": [shared, visual_only]}
        )
        ids = [r.record.evidence_id for r in results]
        assert ids[0] == "shared"

    def test_per_list_rank_populated_for_present_lists(self) -> None:
        r1, r2 = make_record("r1"), make_record("r2")
        results = reciprocal_rank_fusion({"text": [r1, r2], "visual": [r2]})
        r1_result = next(r for r in results if r.record.evidence_id == "r1")
        r2_result = next(r for r in results if r.record.evidence_id == "r2")
        assert r1_result.per_list_rank == {"text": 1}
        assert r2_result.per_list_rank == {"text": 2, "visual": 1}

    def test_absent_from_list_key_not_in_per_list_rank(self) -> None:
        r1 = make_record("r1")
        results = reciprocal_rank_fusion({"text": [r1], "visual": []})
        assert "visual" not in results[0].per_list_rank
        assert results[0].per_list_rank == {"text": 1}

    def test_top_n_limits_results(self) -> None:
        records = [make_record(f"r{i}") for i in range(5)]
        results = reciprocal_rank_fusion({"text": records}, top_n=2)
        assert len(results) == 2

    def test_top_n_none_returns_all(self) -> None:
        records = [make_record(f"r{i}") for i in range(5)]
        results = reciprocal_rank_fusion({"text": records}, top_n=None)
        assert len(results) == 5

    def test_custom_k_changes_scores(self) -> None:
        r1 = make_record("r1")
        results_k60 = reciprocal_rank_fusion({"text": [r1]}, k=60)
        results_k1 = reciprocal_rank_fusion({"text": [r1]}, k=1)
        assert results_k1[0].rrf_score > results_k60[0].rrf_score

    def test_sorted_by_rrf_score_descending(self) -> None:
        r_top = make_record("top")
        r_mid = make_record("mid")
        r_low = make_record("low")
        results = reciprocal_rank_fusion({"text": [r_top, r_mid, r_low], "visual": [r_top]})
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_three_lists_fused_correctly(self) -> None:
        r1 = make_record("r1")
        r2 = make_record("r2")
        r3 = make_record("r3")
        results = reciprocal_rank_fusion({"text": [r1], "caption": [r2], "visual": [r3]})
        assert len(results) == 3
        ids = {r.record.evidence_id for r in results}
        assert ids == {"r1", "r2", "r3"}

    def test_canonical_record_comes_from_highest_ranked_list(self) -> None:
        r_high = make_record("shared")
        r_high_with_caption = EvidenceRecord(
            evidence_id="shared",
            doc_id="doc1",
            source="test.pdf",
            page=1,
            modality="image",
            caption="description from visual list",
        )
        results = reciprocal_rank_fusion(
            {"visual": [r_high_with_caption], "text": [make_record("other"), r_high]}
        )
        shared = next(r for r in results if r.record.evidence_id == "shared")
        assert shared.record.modality == "image"

    def test_source_modality_from_record(self) -> None:
        r_img = make_record("img1", modality="image")
        results = reciprocal_rank_fusion({"visual": [r_img]})
        assert results[0].source_modality == "image"
