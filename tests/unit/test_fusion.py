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


# ===========================================================================
# PR4 — canonical-identity fusion
#
# These tests cover reciprocal_rank_fusion_canonical, which fuses
# RetrievedCandidate lists by canonical evidence identity. The Stage-4 tests
# above (reciprocal_rank_fusion / FusedResult) are unaffected.
# ===========================================================================

from mrta.eval.adapter import EvalAdapter  # noqa: E402
from mrta.eval.types import CanonicalEvidence, RetrievedCandidate  # noqa: E402
from mrta.retrieval.fusion import (  # noqa: E402
    FusedCandidate,
    canonical_identity,
    reciprocal_rank_fusion_canonical,
)

DOC = "doc_attention_2017"
RRF_K = 60


def fig_cand(
    figure_id: str,
    page: int = 3,
    doc: str = DOC,
    score: float = 0.9,
    rank: int = 1,
    candidate_id: str | None = None,
) -> RetrievedCandidate:
    """A figure-evidence candidate."""
    return RetrievedCandidate(
        candidate_id=candidate_id or f"{doc}_p{page}_{figure_id}",
        evidence=CanonicalEvidence(document_id=doc, page_number=page, figure_id=figure_id),
        score=score,
        rank=rank,
    )


def text_cand(
    chunk_id: str,
    page: int = 3,
    doc: str = DOC,
    score: float = 0.9,
    rank: int = 1,
) -> RetrievedCandidate:
    """A text-evidence candidate (figure_id is None)."""
    return RetrievedCandidate(
        candidate_id=chunk_id,
        evidence=CanonicalEvidence(document_id=doc, page_number=page, figure_id=None),
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# RRF arithmetic — exact hand-calculated values
# ---------------------------------------------------------------------------


class TestRRFArithmetic:
    def test_exact_score_single_stream_rank_1(self) -> None:
        fused = reciprocal_rank_fusion_canonical({"text": [text_cand("c1")]}, k=RRF_K)
        assert fused[0].score == pytest.approx(1 / (60 + 1))

    def test_exact_score_two_streams_hand_calculation(self) -> None:
        """Candidate at text rank 2 and caption rank 4 (the spec's worked example)."""
        streams = {
            "text": [text_cand("other1"), text_cand("target")],
            "caption": [
                text_cand("o2"),
                text_cand("o3"),
                text_cand("o4"),
                text_cand("target"),
            ],
        }
        fused = reciprocal_rank_fusion_canonical(streams, k=RRF_K)
        target = next(c for c in fused if c.chunk_id == "target")
        expected = 1 / (60 + 2) + 1 / (60 + 4)
        assert target.score == pytest.approx(expected)

    def test_ranks_are_one_indexed(self) -> None:
        """First list element is rank 1, not rank 0."""
        fused = reciprocal_rank_fusion_canonical({"text": [text_cand("c1")]}, k=RRF_K)
        assert fused[0].source_ranks["text"] == 1
        assert fused[0].score == pytest.approx(1 / 61)  # not 1/60

    def test_three_streams_accumulate(self) -> None:
        c = fig_cand("fig_a")
        streams = {"text": [c], "caption": [c], "clip": [c]}
        fused = reciprocal_rank_fusion_canonical(streams, k=RRF_K)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(3 * (1 / 61))
        assert len(fused[0].modality_sources) == 3

    def test_k_affects_score(self) -> None:
        c = [text_cand("c1")]
        s60 = reciprocal_rank_fusion_canonical({"text": c}, k=60)[0].score
        s1 = reciprocal_rank_fusion_canonical({"text": c}, k=1)[0].score
        assert s1 > s60
        assert s1 == pytest.approx(1 / 2)

    def test_non_positive_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            reciprocal_rank_fusion_canonical({"text": [text_cand("c1")]}, k=0)

    def test_absent_candidate_receives_no_penalty(self) -> None:
        """A candidate missing from a stream contributes 0 — not a negative or bottom rank."""
        present_both = fig_cand("fig_both")
        only_text = text_cand("only_text")
        streams = {"text": [present_both, only_text], "caption": [present_both]}
        fused = reciprocal_rank_fusion_canonical(streams, k=RRF_K)
        solo = next(c for c in fused if c.chunk_id == "only_text")
        # exactly its single text contribution at rank 2, nothing subtracted
        assert solo.score == pytest.approx(1 / 62)
        assert solo.source_ranks == {"text": 2}


# ---------------------------------------------------------------------------
# Stream handling
# ---------------------------------------------------------------------------


class TestStreamHandling:
    def test_single_stream_preserves_ranking(self) -> None:
        """RRF over one stream must not perturb that stream's order (PR4 control A)."""
        ordered = [text_cand(f"c{i}") for i in range(10)]
        fused = reciprocal_rank_fusion_canonical({"text": ordered}, k=RRF_K)
        assert [c.chunk_id for c in fused] == [f"c{i}" for i in range(10)]

    def test_empty_streams_dict(self) -> None:
        assert reciprocal_rank_fusion_canonical({}, k=RRF_K) == []

    def test_all_streams_empty(self) -> None:
        assert reciprocal_rank_fusion_canonical({"text": [], "clip": []}, k=RRF_K) == []

    def test_one_empty_stream_among_others(self) -> None:
        fused = reciprocal_rank_fusion_canonical({"text": [text_cand("c1")], "clip": []}, k=RRF_K)
        assert len(fused) == 1
        assert fused[0].modality_sources == ("text",)

    def test_different_stream_lengths(self) -> None:
        streams = {
            "text": [text_cand(f"t{i}") for i in range(7)],
            "clip": [fig_cand("fig_a")],
        }
        fused = reciprocal_rank_fusion_canonical(streams, k=RRF_K)
        assert len(fused) == 8

    def test_top_k_truncation(self) -> None:
        streams = {"text": [text_cand(f"c{i}") for i in range(10)]}
        assert len(reciprocal_rank_fusion_canonical(streams, k=RRF_K, top_k=3)) == 3

    def test_top_k_none_returns_all(self) -> None:
        streams = {"text": [text_cand(f"c{i}") for i in range(10)]}
        assert len(reciprocal_rank_fusion_canonical(streams, k=RRF_K, top_k=None)) == 10

    def test_candidate_pool_deeper_than_eval_cutoff(self) -> None:
        """Fusion may promote evidence that sat below the eval cutoff in every stream."""
        # target is rank 8 in text and rank 7 in caption — outside a top-5 cut,
        # but its two contributions should lift it above single-stream rank-6 items.
        target = fig_cand("fig_target")
        text_stream = [text_cand(f"t{i}") for i in range(7)] + [target]
        caption_stream = [fig_cand(f"fig_o{i}") for i in range(6)] + [target]
        fused = reciprocal_rank_fusion_canonical(
            {"text": text_stream, "caption": caption_stream}, k=RRF_K, top_k=5
        )
        ids = [c.figure_id for c in fused]
        assert "fig_target" in ids, "two-stream support should promote it into top-5"


# ---------------------------------------------------------------------------
# Canonical identity and deduplication
# ---------------------------------------------------------------------------


class TestCanonicalDeduplication:
    def test_caption_and_clip_same_figure_dedupe(self) -> None:
        """The core PR4 requirement: one physical figure, one fused candidate."""
        cap = fig_cand("fig_siglip_batch_size", page=4, doc="siglip", candidate_id="cap_rec")
        clip = fig_cand("fig_siglip_batch_size", page=4, doc="siglip", candidate_id="clip_rec")
        fused = reciprocal_rank_fusion_canonical({"caption": [cap], "clip": [clip]}, k=RRF_K)
        assert len(fused) == 1

    def test_same_figure_accumulates_both_contributions(self) -> None:
        cap = fig_cand("fig_x", candidate_id="cap_rec")
        clip = fig_cand("fig_x", candidate_id="clip_rec")
        # caption rank 2, clip rank 5
        streams = {
            "caption": [fig_cand("fig_other"), cap],
            "clip": [fig_cand(f"fig_o{i}") for i in range(4)] + [clip],
        }
        fused = reciprocal_rank_fusion_canonical(streams, k=RRF_K)
        target = next(c for c in fused if c.figure_id == "fig_x")
        assert target.score == pytest.approx(1 / (60 + 2) + 1 / (60 + 5))
        assert target.source_ranks == {"caption": 2, "clip": 5}

    def test_modality_provenance_retained(self) -> None:
        c = fig_cand("fig_x")
        fused = reciprocal_rank_fusion_canonical({"caption": [c], "clip": [c]}, k=RRF_K)
        assert set(fused[0].modality_sources) == {"caption", "clip"}

    def test_source_ranks_retained(self) -> None:
        streams = {"text": [text_cand("a"), text_cand("b")], "caption": [text_cand("b")]}
        fused = reciprocal_rank_fusion_canonical(streams, k=RRF_K)
        b = next(c for c in fused if c.chunk_id == "b")
        assert b.source_ranks == {"text": 2, "caption": 1}

    def test_raw_scores_retained_diagnostically(self) -> None:
        c = fig_cand("fig_x", score=0.42)
        fused = reciprocal_rank_fusion_canonical({"clip": [c]}, k=RRF_K)
        assert fused[0].source_scores == {"clip": pytest.approx(0.42)}

    def test_raw_scores_do_not_affect_rrf_score(self) -> None:
        """A wildly higher raw score must not change the fused score."""
        low = reciprocal_rank_fusion_canonical({"clip": [fig_cand("f", score=0.01)]}, k=RRF_K)
        high = reciprocal_rank_fusion_canonical({"clip": [fig_cand("f", score=99.0)]}, k=RRF_K)
        assert low[0].score == pytest.approx(high[0].score)

    def test_two_figures_same_page_stay_separate(self) -> None:
        a = fig_cand("fig_a", page=4)
        b = fig_cand("fig_b", page=4)
        fused = reciprocal_rank_fusion_canonical({"caption": [a, b]}, k=RRF_K)
        assert len(fused) == 2

    def test_text_and_figure_same_page_stay_separate(self) -> None:
        """Hybrid ground truth needs both to be retrievable independently."""
        t = text_cand("chunk_17", page=4)
        f = fig_cand("fig_2", page=4)
        fused = reciprocal_rank_fusion_canonical({"text": [t], "caption": [f]}, k=RRF_K)
        assert len(fused) == 2
        assert {c.evidence_type for c in fused} == {"text", "figure"}

    def test_two_text_chunks_same_page_stay_separate(self) -> None:
        """CanonicalEvidence alone cannot distinguish these — chunk_id must."""
        c1 = text_cand("doc_p4_c1", page=4)
        c2 = text_cand("doc_p4_c2", page=4)
        fused = reciprocal_rank_fusion_canonical({"text": [c1, c2]}, k=RRF_K)
        assert len(fused) == 2

    def test_duplicate_within_one_stream_counted_once(self) -> None:
        """Several crops of one figure in a single stream must not inflate its score."""
        crop1 = fig_cand("fig_multi", candidate_id="crop1")
        crop2 = fig_cand("fig_multi", candidate_id="crop2")
        crop3 = fig_cand("fig_multi", candidate_id="crop3")
        fused = reciprocal_rank_fusion_canonical({"caption": [crop1, crop2, crop3]}, k=RRF_K)
        assert len(fused) == 1
        # only the best (first) occurrence contributes
        assert fused[0].score == pytest.approx(1 / 61)
        assert fused[0].source_ranks == {"caption": 1}

    def test_no_duplicate_canonical_ids_in_output(self) -> None:
        streams = {
            "text": [text_cand("a"), text_cand("b")],
            "caption": [fig_cand("f1"), fig_cand("f1", candidate_id="other")],
            "clip": [fig_cand("f1", candidate_id="third"), text_cand("a")],
        }
        fused = reciprocal_rank_fusion_canonical(streams, k=RRF_K)
        ids = [c.canonical_id for c in fused]
        assert len(ids) == len(set(ids))

    def test_canonical_id_is_stable(self) -> None:
        c = fig_cand("fig_x", page=7, doc="clip")
        first = reciprocal_rank_fusion_canonical({"caption": [c]}, k=RRF_K)[0].canonical_id
        second = reciprocal_rank_fusion_canonical({"clip": [c]}, k=RRF_K)[0].canonical_id
        assert first == second

    def test_canonical_identity_figure_shape(self) -> None:
        assert canonical_identity(fig_cand("fig_x", page=3, doc="d")) == (
            "figure",
            "d",
            "p3:fig_x",
        )

    def test_canonical_identity_text_shape(self) -> None:
        assert canonical_identity(text_cand("chunk_9", doc="d")) == ("text", "d", "chunk_9")


# ---------------------------------------------------------------------------
# Payload merging
# ---------------------------------------------------------------------------


class TestPayloadMerge:
    def test_payload_merges_across_streams(self) -> None:
        cap = fig_cand("fig_x")
        clip = fig_cand("fig_x", candidate_id="clip_rec")
        cid = canonical_identity(cap)
        cid_str = "|".join(cid)
        payloads = {
            "caption": {cid_str: {"caption_text": "a diagram"}},
            "clip": {cid_str: {"image_path": "data/x.png"}},
        }
        fused = reciprocal_rank_fusion_canonical(
            {"caption": [cap], "clip": [clip]}, k=RRF_K, payloads=payloads
        )
        assert fused[0].payload == {"caption_text": "a diagram", "image_path": "data/x.png"}

    def test_payload_conflict_is_deterministic(self) -> None:
        """Same inputs in different dict orders must yield the same payload."""
        cap = fig_cand("fig_x")
        clip = fig_cand("fig_x", candidate_id="clip_rec")
        cid_str = "|".join(canonical_identity(cap))
        p_a = {
            "caption": {cid_str: {"image_path": "from_caption.png"}},
            "clip": {cid_str: {"image_path": "from_clip.png"}},
        }
        p_b = {
            "clip": {cid_str: {"image_path": "from_clip.png"}},
            "caption": {cid_str: {"image_path": "from_caption.png"}},
        }
        r_a = reciprocal_rank_fusion_canonical(
            {"caption": [cap], "clip": [clip]}, k=RRF_K, payloads=p_a
        )
        r_b = reciprocal_rank_fusion_canonical(
            {"clip": [clip], "caption": [cap]}, k=RRF_K, payloads=p_b
        )
        assert r_a[0].payload == r_b[0].payload

    def test_payload_does_not_affect_score(self) -> None:
        c = fig_cand("fig_x")
        cid_str = "|".join(canonical_identity(c))
        with_payload = reciprocal_rank_fusion_canonical(
            {"caption": [c]}, k=RRF_K, payloads={"caption": {cid_str: {"k": "v"}}}
        )
        without = reciprocal_rank_fusion_canonical({"caption": [c]}, k=RRF_K)
        assert with_payload[0].score == pytest.approx(without[0].score)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_tie_break_prefers_better_source_rank(self) -> None:
        """Equal RRF scores -> the candidate with the better best-rank wins."""
        # Both appear once at rank 1 in different streams => identical scores.
        a = fig_cand("fig_bbb")
        b = fig_cand("fig_aaa")
        fused = reciprocal_rank_fusion_canonical({"caption": [a], "clip": [b]}, k=RRF_K)
        assert fused[0].score == pytest.approx(fused[1].score)
        # tie falls through to canonical_id ascending: fig_aaa before fig_bbb
        assert fused[0].figure_id == "fig_aaa"

    def test_tie_break_prefers_more_streams(self) -> None:
        """Two candidates, equal best rank; more contributing streams ranks higher."""
        multi = fig_cand("fig_multi")
        single = fig_cand("fig_single")
        # multi: rank 1 in two streams; single: rank 1 in one stream
        fused = reciprocal_rank_fusion_canonical(
            {"caption": [multi], "clip": [multi], "text": [single]}, k=RRF_K
        )
        assert fused[0].figure_id == "fig_multi"
        assert len(fused[0].modality_sources) == 2

    def test_stream_dict_ordering_does_not_change_ranking(self) -> None:
        a, b, c = fig_cand("fig_a"), fig_cand("fig_b"), fig_cand("fig_c")
        order_1 = {"text": [a], "caption": [b], "clip": [c]}
        order_2 = {"clip": [c], "text": [a], "caption": [b]}
        r1 = [x.canonical_id for x in reciprocal_rank_fusion_canonical(order_1, k=RRF_K)]
        r2 = [x.canonical_id for x in reciprocal_rank_fusion_canonical(order_2, k=RRF_K)]
        assert r1 == r2

    def test_repeated_runs_are_identical(self) -> None:
        streams = {
            "text": [text_cand(f"t{i}") for i in range(5)],
            "caption": [fig_cand(f"f{i}") for i in range(5)],
        }
        runs = [
            [c.canonical_id for c in reciprocal_rank_fusion_canonical(streams, k=RRF_K)]
            for _ in range(5)
        ]
        assert all(r == runs[0] for r in runs)


# ---------------------------------------------------------------------------
# EvalAdapter integration
# ---------------------------------------------------------------------------

_MANIFEST = {
    "documents": [
        {
            "document_id": DOC,
            "filename": "attention_is_all_you_need.pdf",
            "figures": [
                {"figure_id": "fig_transformer_arch", "page_number": 3, "figure_index": 1},
            ],
        }
    ]
}


class TestAdapterConversion:
    def test_figure_maps_to_evaluation_evidence(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        fused = reciprocal_rank_fusion_canonical(
            {"caption": [fig_cand("fig_transformer_arch", page=3)]}, k=RRF_K
        )
        rc = adapter.from_fused_candidate(fused[0], rank=1)
        assert rc.evidence.document_id == DOC
        assert rc.evidence.page_number == 3
        assert rc.evidence.figure_id == "fig_transformer_arch"

    def test_figure_matches_its_target(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        fused = reciprocal_rank_fusion_canonical(
            {"caption": [fig_cand("fig_transformer_arch", page=3)]}, k=RRF_K
        )
        rc = adapter.from_fused_candidate(fused[0], rank=1)
        target = CanonicalEvidence(DOC, 3, "fig_transformer_arch")
        assert target.matches(rc.evidence)

    def test_text_maps_to_evaluation_evidence(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        fused = reciprocal_rank_fusion_canonical({"text": [text_cand("c1", page=2)]}, k=RRF_K)
        rc = adapter.from_fused_candidate(fused[0], rank=1)
        assert rc.evidence.figure_id is None
        assert rc.evidence.page_number == 2
        assert CanonicalEvidence(DOC, 2, None).matches(rc.evidence)

    def test_wrong_figure_does_not_match(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        fused = reciprocal_rank_fusion_canonical(
            {"caption": [fig_cand("fig_other", page=3)]}, k=RRF_K
        )
        rc = adapter.from_fused_candidate(fused[0], rank=1)
        assert not CanonicalEvidence(DOC, 3, "fig_transformer_arch").matches(rc.evidence)

    def test_rank_is_carried_through(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        fused = reciprocal_rank_fusion_canonical({"text": [text_cand("c1")]}, k=RRF_K)
        assert adapter.from_fused_candidate(fused[0], rank=4).rank == 4

    def test_fused_candidate_is_frozen(self) -> None:
        fused = reciprocal_rank_fusion_canonical({"text": [text_cand("c1")]}, k=RRF_K)
        with pytest.raises(Exception):
            fused[0].score = 1.0  # type: ignore[misc]

    def test_returns_fused_candidate_instances(self) -> None:
        fused = reciprocal_rank_fusion_canonical({"text": [text_cand("c1")]}, k=RRF_K)
        assert all(isinstance(c, FusedCandidate) for c in fused)
