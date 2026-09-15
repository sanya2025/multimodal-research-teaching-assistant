"""Unit tests for the PR7 ablation framework.

Fully mocked: no Ollama, no GPU, no network, no model downloads.
"""

from __future__ import annotations

import json

import pytest

from mrta.core.schemas import Chunk
from mrta.eval.ablation import (
    CONFIGURATIONS_BY_ID,
    FAILURE_ANSWER_SUPPORT,
    FAILURE_CITATION_MISSING,
    FAILURE_CITATION_VALIDITY,
    FAILURE_NONE,
    FAILURE_RANKING_MISS,
    FAILURE_RETRIEVAL_MISS,
    FROZEN_CONFIGURATIONS,
    HISTORICAL_CONFIGURATIONS,
    STATUS_ERROR,
    STATUS_OK,
    AblationConfig,
    QueryResult,
    aggregate,
    classify_failure,
    compute_retrieval_metrics,
    figure_provenance,
    parse_targets,
    percentile,
    target_rank,
)
from mrta.eval.ablation_runner import AblationRunner, RunnerStores, build_figure_text_lookup
from mrta.eval.types import CanonicalEvidence, RetrievedCandidate

DOC = "attention_is_all_you_need"


def candidate(page: int, figure_id: str | None = None, rank: int = 1, cid: str | None = None):
    return RetrievedCandidate(
        candidate_id=cid or f"c_{page}_{figure_id}",
        evidence=CanonicalEvidence(document_id=DOC, page_number=page, figure_id=figure_id),
        score=1.0 / rank,
        rank=rank,
    )


def target(page: int, figure_id: str | None = None) -> CanonicalEvidence:
    return CanonicalEvidence(document_id=DOC, page_number=page, figure_id=figure_id)


# ---------------------------------------------------------------------------
# Configuration parsing and validation
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_frozen_matrix_has_expected_members(self) -> None:
        ids = {c.config_id for c in FROZEN_CONFIGURATIONS}
        assert {
            "text_only",
            "caption_only",
            "clip_only",
            "text_caption_rrf",
            "text_clip_rrf",
            "caption_clip_rrf",
            "text_caption_clip_rrf",
            "text_caption_rrf_reranked",
            "full_reranked",
            "oracle_evidence_generation",
        } == ids

    def test_historical_configs_all_exist(self) -> None:
        for config_id in HISTORICAL_CONFIGURATIONS:
            assert config_id in CONFIGURATIONS_BY_ID

    def test_single_stream_cannot_be_fused(self) -> None:
        with pytest.raises(ValueError, match="fusion needs 2"):
            AblationConfig("bad", ("text",), fuse=True)

    def test_rerank_requires_fusion(self) -> None:
        with pytest.raises(ValueError, match="fused candidate pool"):
            AblationConfig("bad", ("text",), rerank=True)

    def test_unknown_stream_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown stream"):
            AblationConfig("bad", ("audio",))

    def test_empty_streams_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one stream"):
            AblationConfig("bad", ())

    def test_oracle_cannot_combine_with_retrieval(self) -> None:
        with pytest.raises(ValueError, match="bypasses retrieval"):
            AblationConfig("bad", ("text",), oracle_evidence=True)

    def test_oracle_is_not_a_retrieval_config(self) -> None:
        assert CONFIGURATIONS_BY_ID["oracle_evidence_generation"].is_retrieval is False
        assert CONFIGURATIONS_BY_ID["full_reranked"].is_retrieval is True

    def test_config_file_matches_frozen_matrix(self) -> None:
        """configs/ablation_config.yaml must not drift from the code matrix."""
        import pathlib

        import yaml

        path = pathlib.Path(__file__).resolve().parents[2] / "configs/ablation_config.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(cfg["configurations"]) == set(CONFIGURATIONS_BY_ID)
        assert cfg["candidate_depth"] == 20
        assert cfg["rrf_k"] == 60
        assert cfg["final_top_k"] == 5


# ---------------------------------------------------------------------------
# Metrics and helpers
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    def test_perfect_retrieval(self) -> None:
        m = compute_retrieval_metrics([candidate(2)], [target(2)])
        assert m["recall_at_1"] == 1.0
        assert m["mrr_at_5"] == 1.0

    def test_metrics_use_top_k_slice_only(self) -> None:
        """A target at rank 6 must not contribute to MRR@5."""
        candidates = [candidate(99, rank=i) for i in range(1, 6)] + [candidate(2, rank=6)]
        m = compute_retrieval_metrics(candidates, [target(2)], final_top_k=5)
        assert m["recall_at_5"] == 0.0
        assert m["mrr_at_5"] == 0.0

    def test_figure_recall_none_without_figure_targets(self) -> None:
        m = compute_retrieval_metrics([candidate(2)], [target(2)])
        assert m["figure_recall_at_5"] is None

    def test_hybrid_both_target_none_for_text_only_query(self) -> None:
        m = compute_retrieval_metrics([candidate(2)], [target(2)])
        assert m["both_target_recall_at_5"] is None

    def test_hybrid_both_target_true(self) -> None:
        cands = [candidate(2), candidate(4, "p4_f1", rank=2)]
        m = compute_retrieval_metrics(cands, [target(2), target(4, "p4_f1")])
        assert m["both_target_recall_at_5"] is True

    def test_target_rank(self) -> None:
        cands = [candidate(9, rank=1), candidate(2, rank=2)]
        assert target_rank(cands, [target(2)]) == 2
        assert target_rank(cands, [target(77)]) is None

    def test_percentile_nearest_rank(self) -> None:
        assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
        assert percentile([], 95) is None

    def test_parse_targets_supports_both_schemas(self) -> None:
        v2 = {"expected_evidence": [{"document_id": DOC, "page_number": 3, "figure_id": None}]}
        v1 = {"target_evidence": [{"document_id": DOC, "page_number": 3}]}
        assert parse_targets(v2)[0].page_number == 3
        assert parse_targets(v1)[0].page_number == 3


class TestFigureProvenance:
    def test_vlm_caption(self) -> None:
        assert figure_provenance({"figure_caption": "A diagram."}) == "vlm_caption"

    def test_nearby_text_fallback(self) -> None:
        assert figure_provenance({"figure_nearby_text": "text"}) == "nearby_text_fallback"

    def test_caption_wins_over_fallback(self) -> None:
        payload = {"figure_caption": "A.", "figure_nearby_text": "B."}
        assert figure_provenance(payload) == "vlm_caption"

    def test_empty_payload(self) -> None:
        assert figure_provenance({}) is None
        assert figure_provenance(None) is None


# ---------------------------------------------------------------------------
# Failure attribution
# ---------------------------------------------------------------------------


class TestFailureAttribution:
    def test_retrieval_miss_when_target_never_in_pool(self) -> None:
        assert (
            classify_failure(
                targets=[target(2)],
                pool_rank=None,
                final_rank=None,
                generation_ran=False,
                citation_scores=None,
                validity_rate=None,
                support_score=None,
            )
            == FAILURE_RETRIEVAL_MISS
        )

    def test_ranking_miss_when_in_pool_but_not_final(self) -> None:
        assert (
            classify_failure(
                targets=[target(2)],
                pool_rank=12,
                final_rank=None,
                generation_ran=False,
                citation_scores=None,
                validity_rate=None,
                support_score=None,
            )
            == FAILURE_RANKING_MISS
        )

    def test_retrieval_failure_not_blamed_on_generation(self) -> None:
        """Earliest-stage attribution: a missing target is not a citation failure."""
        assert (
            classify_failure(
                targets=[target(2)],
                pool_rank=None,
                final_rank=None,
                generation_ran=True,
                citation_scores={"citation_recall": 0.0, "citation_precision": 0.0},
                validity_rate=1.0,
                support_score=1.0,
            )
            == FAILURE_RETRIEVAL_MISS
        )

    def test_citation_missing(self) -> None:
        assert (
            classify_failure(
                targets=[target(2)],
                pool_rank=1,
                final_rank=1,
                generation_ran=True,
                citation_scores={"citation_recall": 0.0, "citation_precision": 1.0},
                validity_rate=1.0,
                support_score=1.0,
            )
            == FAILURE_CITATION_MISSING
        )

    def test_citation_validity_takes_precedence(self) -> None:
        assert (
            classify_failure(
                targets=[target(2)],
                pool_rank=1,
                final_rank=1,
                generation_ran=True,
                citation_scores={"citation_recall": 1.0, "citation_precision": 1.0},
                validity_rate=0.5,
                support_score=1.0,
            )
            == FAILURE_CITATION_VALIDITY
        )

    def test_answer_support_failure(self) -> None:
        assert (
            classify_failure(
                targets=[target(2)],
                pool_rank=1,
                final_rank=1,
                generation_ran=True,
                citation_scores={"citation_recall": 1.0, "citation_precision": 1.0},
                validity_rate=1.0,
                support_score=0.1,
            )
            == FAILURE_ANSWER_SUPPORT
        )

    def test_no_failure(self) -> None:
        assert (
            classify_failure(
                targets=[target(2)],
                pool_rank=1,
                final_rank=1,
                generation_ran=True,
                citation_scores={"citation_recall": 1.0, "citation_precision": 1.0},
                validity_rate=1.0,
                support_score=1.0,
            )
            == FAILURE_NONE
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def _row(self, recall: float, status: str = STATUS_OK) -> QueryResult:
        row = QueryResult(query_id="q", config_id="c", status=status)
        row.retrieval_metrics = {"recall_at_5": recall, "mrr_at_5": recall}
        return row

    def test_means_over_successful_rows(self) -> None:
        summary = aggregate([self._row(1.0), self._row(0.0)])
        assert summary["recall_at_5"] == pytest.approx(0.5)
        assert summary["sample_count"] == 2

    def test_failed_rows_excluded_from_means_but_counted(self) -> None:
        """A crashed query must not quietly improve an average."""
        summary = aggregate([self._row(1.0), self._row(0.0, status=STATUS_ERROR)])
        assert summary["recall_at_5"] == pytest.approx(1.0)
        assert summary["failed_count"] == 1
        assert summary["failure_rate"] == pytest.approx(0.5)
        assert summary["successful_count"] == 1

    def test_empty_rows(self) -> None:
        summary = aggregate([])
        assert summary["sample_count"] == 0
        assert summary["failure_rate"] == 0.0

    def test_all_failed_reports_no_metrics(self) -> None:
        summary = aggregate([self._row(1.0, status=STATUS_ERROR)])
        assert summary["failed_count"] == 1
        assert "recall_at_5" not in summary

    def test_none_metrics_excluded_from_mean(self) -> None:
        a, b = self._row(1.0), self._row(1.0)
        a.retrieval_metrics["figure_recall_at_5"] = None
        b.retrieval_metrics["figure_recall_at_5"] = 0.5
        summary = aggregate([a, b])
        assert summary["figure_recall_at_5"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class FakeAdapter:
    """EvalAdapter stand-in mapping fake records to canonical evidence."""

    def chunk_to_candidate(self, chunk, score, rank):
        return RetrievedCandidate(
            candidate_id=chunk.chunk_id,
            evidence=CanonicalEvidence(DOC, chunk.page, None),
            score=score,
            rank=rank,
        )

    def from_caption_record(self, record, score, rank):
        return RetrievedCandidate(
            candidate_id=record.evidence_id,
            evidence=CanonicalEvidence(DOC, record.page, record.figure_id),
            score=score,
            rank=rank,
        )

    def from_visual_record(self, record, score, rank):
        return RetrievedCandidate(
            candidate_id=record.record_id,
            evidence=CanonicalEvidence(DOC, record.page, record.figure_id),
            score=score,
            rank=rank,
        )


class FakeRecord:
    def __init__(self, page, figure_id, caption=None, nearby=None, evidence_id=None):
        self.page = page
        self.figure_id = figure_id
        self.caption = caption
        self.detailed_description = None
        self.nearby_text = nearby
        self.evidence_id = evidence_id or f"e_{page}_{figure_id}"
        self.record_id = self.evidence_id


class FakeStore:
    def __init__(self, hits, records=None):
        self._hits = hits
        self._records = records or []

    def search_with_scores(self, query, k=5):
        return self._hits[:k]

    def search(self, query, top_k=5):
        return self._hits[:top_k]


def _chunk(chunk_id: str, page: int, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=DOC, source="attention.pdf", page=page, text=text)


def _text_store():
    return FakeStore(
        [
            (_chunk("c1", 2, "Attention uses softmax scaling."), 0.9),
            (_chunk("c2", 7, "Unrelated content here."), 0.5),
        ]
    )


def _caption_store():
    record = FakeRecord(4, "fig_arch", caption="Diagram of the architecture.")
    return FakeStore([(record, 0.8)], records=[record])


class TestRunner:
    def _runner(self, **kw):
        stores = RunnerStores(
            text=_text_store(),
            caption=_caption_store(),
            adapter=FakeAdapter(),
            **kw,
        )
        return AblationRunner(stores)

    def test_adapter_is_required(self) -> None:
        runner = AblationRunner(RunnerStores(text=_text_store()))
        with pytest.raises(RuntimeError, match="EvalAdapter"):
            runner.retrieve_streams("q")

    def test_retrieval_only_mode_produces_metrics(self) -> None:
        runner = self._runner()
        pools, latency = runner.retrieve_streams("attention softmax")
        query = {
            "query_id": "q1",
            "query": "attention softmax",
            "expected_evidence": [{"document_id": DOC, "page_number": 2, "figure_id": None}],
        }
        result = runner.run_configuration(CONFIGURATIONS_BY_ID["text_only"], query, pools, latency)
        assert result.status == STATUS_OK
        assert result.retrieval_metrics["recall_at_5"] == 1.0
        assert result.generated_answer is None

    def test_missing_stream_is_recorded_not_raised(self) -> None:
        """One unavailable stream must not end the run."""
        runner = AblationRunner(RunnerStores(text=_text_store(), adapter=FakeAdapter()))
        pools, latency = runner.retrieve_streams("q")
        query = {"query_id": "q1", "query": "q", "expected_evidence": []}
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["text_caption_rrf"], query, pools, latency
        )
        assert result.status == STATUS_ERROR
        assert "unavailable" in (result.error_message or "")

    def test_fusion_configuration_runs(self) -> None:
        runner = self._runner()
        pools, latency = runner.retrieve_streams("attention")
        query = {
            "query_id": "q1",
            "query": "attention",
            "expected_evidence": [{"document_id": DOC, "page_number": 4, "figure_id": "fig_arch"}],
        }
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["text_caption_rrf"], query, pools, latency
        )
        assert result.status == STATUS_OK
        assert result.latency_ms["fusion"] >= 0

    def test_oracle_mode_has_perfect_retrieval_by_construction(self) -> None:
        runner = self._runner()
        query = {
            "query_id": "q1",
            "query": "anything",
            "expected_evidence": [{"document_id": DOC, "page_number": 2, "figure_id": None}],
        }
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["oracle_evidence_generation"], query, {}, {}
        )
        assert result.retrieval_metrics["recall_at_5"] == 1.0
        assert result.retrieval_metrics["mrr_at_5"] == 1.0

    def test_generation_with_mocked_generator(self) -> None:
        class FakeGenerator:
            def generate(self, prompt, images):
                return "Attention uses softmax scaling [T1]."

        stores = RunnerStores(text=_text_store(), caption=_caption_store(), adapter=FakeAdapter())
        runner = AblationRunner(stores, generator=FakeGenerator())
        pools, latency = runner.retrieve_streams("attention softmax")
        query = {
            "query_id": "q1",
            "query": "attention softmax",
            "expected_evidence": [{"document_id": DOC, "page_number": 2, "figure_id": None}],
        }
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["text_only"], query, pools, latency, generate=True
        )
        assert result.generated_answer
        assert result.generation_metrics["citation_recall"] == 1.0
        assert result.coverage["text_target_covered"] is True
        assert result.latency_ms["generation"] >= 0

    def test_generator_failure_is_captured_as_error(self) -> None:
        class ExplodingGenerator:
            def generate(self, prompt, images):
                raise RuntimeError("model unavailable")

        stores = RunnerStores(text=_text_store(), adapter=FakeAdapter())
        runner = AblationRunner(stores, generator=ExplodingGenerator())
        pools, latency = runner.retrieve_streams("q")
        query = {"query_id": "q1", "query": "q", "expected_evidence": []}
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["text_only"], query, pools, latency, generate=True
        )
        assert result.status == STATUS_ERROR
        assert result.error_type == "RuntimeError"

    def test_invalid_citation_label_is_flagged(self) -> None:
        class PhantomGenerator:
            def generate(self, prompt, images):
                return "See [T9] for details."

        stores = RunnerStores(text=_text_store(), adapter=FakeAdapter())
        runner = AblationRunner(stores, generator=PhantomGenerator())
        pools, latency = runner.retrieve_streams("q")
        query = {
            "query_id": "q1",
            "query": "q",
            "expected_evidence": [{"document_id": DOC, "page_number": 2, "figure_id": None}],
        }
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["text_only"], query, pools, latency, generate=True
        )
        assert result.generation_metrics["invalid_citation_count"] >= 1
        assert result.generation_metrics["citation_validity_rate"] < 1.0

    def test_deterministic_across_repeated_runs(self) -> None:
        query = {
            "query_id": "q1",
            "query": "attention",
            "expected_evidence": [{"document_id": DOC, "page_number": 2, "figure_id": None}],
        }
        seen = []
        for _ in range(3):
            runner = self._runner()
            pools, latency = runner.retrieve_streams("attention")
            result = runner.run_configuration(
                CONFIGURATIONS_BY_ID["text_caption_rrf"], query, pools, latency
            )
            seen.append(json.dumps(result.retrieval_metrics, sort_keys=True))
        assert len(set(seen)) == 1

    def test_row_count_is_queries_times_configs(self) -> None:
        runner = self._runner()
        pools, latency = runner.retrieve_streams("attention")
        queries = [
            {"query_id": f"q{i}", "query": "attention", "expected_evidence": []} for i in range(3)
        ]
        configs = [CONFIGURATIONS_BY_ID[c] for c in ("text_only", "caption_only")]
        rows = [runner.run_configuration(c, q, pools, latency) for q in queries for c in configs]
        assert len(rows) == 6


class TestFigureTextLookup:
    def test_prefers_vlm_caption_over_fallback(self) -> None:
        """Multi-crop resolution must not depend on which crop search returned."""
        fallback = FakeRecord(4, "fig_arch", nearby="page text", evidence_id="e_a")
        captioned = FakeRecord(4, "fig_arch", caption="A diagram.", evidence_id="e_b")
        store = FakeStore([], records=[fallback, captioned])
        lookup = build_figure_text_lookup(store, FakeAdapter())
        assert len(lookup) == 1
        assert next(iter(lookup.values()))["figure_caption"] == "A diagram."

    def test_ties_broken_by_lowest_evidence_id(self) -> None:
        second = FakeRecord(4, "fig_arch", caption="Second.", evidence_id="e_z")
        first = FakeRecord(4, "fig_arch", caption="First.", evidence_id="e_a")
        lookup = build_figure_text_lookup(FakeStore([], records=[second, first]), FakeAdapter())
        assert next(iter(lookup.values()))["figure_caption"] == "First."

    def test_text_records_excluded(self) -> None:
        text_record = FakeRecord(2, None, caption=None)
        assert build_figure_text_lookup(FakeStore([], [text_record]), FakeAdapter()) == {}


class TestBenchmarkIntegrity:
    def test_v2_benchmark_unchanged(self) -> None:
        """PR7 must not have modified the frozen benchmark."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        dataset = json.loads((root / "data/eval/queries_v2.json").read_text(encoding="utf-8"))
        assert dataset["dataset_version"] == "v2.0.0"
        assert dataset["query_count"] == 100
        assert len(dataset["queries"]) == 100

    def test_no_hardcoded_scores_in_ablation_module(self) -> None:
        """Metric values must come from execution, never from a literal."""
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2] / "src/mrta/eval/ablation.py"
        ).read_text(encoding="utf-8")
        for frozen_value in ("0.6050", "0.5153", "0.3292", "0.5467"):
            assert frozen_value not in source


class TestOracleInvariants:
    """The oracle must supply exactly the expected evidence, and nothing else.

    These guard the interpretation, not just the code: if the oracle silently
    dropped a target (a hybrid query's figure, say) its citation recall would be
    measured against a context that never contained the evidence, and the
    "generation ceiling" reading of the result would be wrong.
    """

    def _runner(self):
        stores = RunnerStores(text=_text_store(), caption=_caption_store(), adapter=FakeAdapter())
        return AblationRunner(stores)

    def _oracle(self, expected):
        runner = self._runner()
        query = {"query_id": "q1", "query": "q", "expected_evidence": expected}
        return runner.run_configuration(
            CONFIGURATIONS_BY_ID["oracle_evidence_generation"], query, {}, {}
        )

    def test_oracle_context_equals_expected_evidence(self) -> None:
        expected = [
            {"document_id": DOC, "page_number": 3, "figure_id": None},
            {"document_id": DOC, "page_number": 4, "figure_id": "fig_arch"},
        ]
        result = self._oracle(expected)
        assert result.final_evidence == expected

    def test_oracle_retrieval_recall_is_one(self) -> None:
        result = self._oracle([{"document_id": DOC, "page_number": 3, "figure_id": None}])
        assert result.retrieval_metrics["recall_at_5"] == 1.0

    def test_oracle_supplies_both_kinds_for_hybrid_targets(self) -> None:
        """A hybrid query must reach the generator with text AND figure evidence."""
        result = self._oracle(
            [
                {"document_id": DOC, "page_number": 3, "figure_id": None},
                {"document_id": DOC, "page_number": 4, "figure_id": "fig_arch"},
            ]
        )
        assert any(e["figure_id"] is None for e in result.final_evidence)
        assert any(e["figure_id"] is not None for e in result.final_evidence)

    def test_oracle_citation_recall_is_not_retrieval_recall(self) -> None:
        """Citation recall must measure citing behaviour, not evidence availability.

        The generator is given perfect evidence but cites none of it: retrieval
        recall stays 1.0 while citation recall must fall to 0.0. If the two were
        wired to the same computation this would be impossible.
        """

        class SilentGenerator:
            def generate(self, prompt, images):
                return "An answer with no citations at all."

        stores = RunnerStores(text=_text_store(), adapter=FakeAdapter())
        runner = AblationRunner(stores, generator=SilentGenerator())
        query = {
            "query_id": "q1",
            "query": "q",
            "expected_evidence": [{"document_id": DOC, "page_number": 3, "figure_id": None}],
        }
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["oracle_evidence_generation"], query, {}, {}, generate=True
        )
        assert result.retrieval_metrics["recall_at_5"] == 1.0
        assert result.generation_metrics["citation_recall"] == 0.0


class TestTargetProvenance:
    """Target-figure provenance is PR5's grouping; retrieved-figure is not."""

    def test_target_provenance_uses_expected_figure_not_retrieved(self) -> None:
        captioned = FakeRecord(4, "fig_arch", caption="A diagram.", evidence_id="e_arch")
        fallback = FakeRecord(9, "fig_other", nearby="page text", evidence_id="e_other")
        stores = RunnerStores(
            text=_text_store(),
            caption=FakeStore([(captioned, 0.9)], records=[captioned, fallback]),
            adapter=FakeAdapter(),
        )
        runner = AblationRunner(stores)
        pools, latency = runner.retrieve_streams("q")
        # expected target is the FALLBACK figure, while retrieval surfaces the captioned one
        query = {
            "query_id": "q1",
            "query": "q",
            "expected_evidence": [{"document_id": DOC, "page_number": 9, "figure_id": "fig_other"}],
        }
        result = runner.run_configuration(
            CONFIGURATIONS_BY_ID["caption_only"], query, pools, latency
        )
        assert result.target_figure_provenance == "nearby_text_fallback"
        assert result.top_retrieved_figure_provenance == "vlm_caption"

    def test_target_provenance_none_for_text_only_query(self) -> None:
        stores = RunnerStores(text=_text_store(), caption=_caption_store(), adapter=FakeAdapter())
        runner = AblationRunner(stores)
        pools, latency = runner.retrieve_streams("q")
        query = {
            "query_id": "q1",
            "query": "q",
            "expected_evidence": [{"document_id": DOC, "page_number": 2, "figure_id": None}],
        }
        result = runner.run_configuration(CONFIGURATIONS_BY_ID["text_only"], query, pools, latency)
        assert result.target_figure_provenance is None


class TestPR8ExperimentalIntegrity:
    """PR8's causal claim requires evidence held fixed across conditions."""

    def _stores(self):
        return RunnerStores(text=_text_store(), caption=_caption_store(), adapter=FakeAdapter())

    def _query(self):
        return {
            "query_id": "q1",
            "query": "attention softmax",
            "expected_evidence": [{"document_id": DOC, "page_number": 2, "figure_id": None}],
        }

    def _run_all_conditions(self, generator_factory):
        from mrta.eval.generation_conditions import FROZEN_CONDITIONS

        stores = self._stores()
        primary = AblationRunner(stores, generator=generator_factory())
        pools, latency = primary.retrieve_streams("attention softmax")

        results = {}
        for condition in FROZEN_CONDITIONS:
            runner = AblationRunner(stores, generator=generator_factory(), condition=condition)
            runner.adopt_retrieval_state(primary)
            results[condition.condition_id] = runner.run_configuration(
                CONFIGURATIONS_BY_ID["text_caption_rrf"],
                self._query(),
                pools,
                latency,
                generate=True,
            )
        return results

    def test_all_conditions_receive_identical_evidence(self) -> None:
        class Gen:
            def generate(self, prompt, images):
                return "An answer [T1]."

        results = self._run_all_conditions(Gen)
        hashes = {r.evidence_context_hash for r in results.values()}
        assert len(hashes) == 1, f"evidence diverged across conditions: {hashes}"
        assert all(r.status == STATUS_OK for r in results.values())

    def test_each_condition_is_labelled(self) -> None:
        class Gen:
            def generate(self, prompt, images):
                return "An answer [T1]."

        results = self._run_all_conditions(Gen)
        assert set(results) == {r.generation_condition for r in results.values()}

    def test_conditions_receive_different_prompts(self) -> None:
        """Same evidence, different prompt — that is the whole experiment."""
        from mrta.eval.generation_conditions import FROZEN_CONDITIONS

        class RecordingGen:
            prompts: list[str] = []

            def generate(self, prompt, images):
                RecordingGen.prompts.append(prompt)
                return "An answer [T1]."

        RecordingGen.prompts = []
        stores = self._stores()
        primary = AblationRunner(stores, generator=RecordingGen())
        pools, latency = primary.retrieve_streams("attention softmax")
        for condition in FROZEN_CONDITIONS:
            runner = AblationRunner(stores, generator=RecordingGen(), condition=condition)
            runner.adopt_retrieval_state(primary)
            runner.run_configuration(
                CONFIGURATIONS_BY_ID["text_caption_rrf"],
                self._query(),
                pools,
                latency,
                generate=True,
            )
        assert len(set(RecordingGen.prompts)) == len(FROZEN_CONDITIONS)

    def test_adopt_retrieval_state_does_not_rerun_retrieval(self) -> None:
        """Sharing state must not touch the stores again."""

        class CountingStore(FakeStore):
            def __init__(self, hits, records=None):
                super().__init__(hits, records)
                self.search_count = 0

            def search_with_scores(self, query, k=5):
                self.search_count += 1
                return super().search_with_scores(query, k)

        text = CountingStore(
            [
                (_chunk("c1", 2, "Attention uses softmax scaling."), 0.9),
                (_chunk("c2", 7, "Unrelated content here."), 0.5),
            ]
        )
        stores = RunnerStores(text=text, caption=_caption_store(), adapter=FakeAdapter())
        primary = AblationRunner(stores)
        primary.retrieve_streams("attention softmax")
        assert text.search_count == 1

        secondary = AblationRunner(stores, condition=CONFIGURATIONS_BY_ID["text_only"])
        secondary.adopt_retrieval_state(primary)
        assert text.search_count == 1  # unchanged: no second retrieval

    def test_retrieval_parameters_unchanged_from_pr5(self) -> None:
        """PR8 freezes retrieval; these are the PR4/PR5 evaluated values."""
        from mrta.eval.ablation import (
            DEFAULT_CANDIDATE_DEPTH,
            DEFAULT_FINAL_TOP_K,
            DEFAULT_RRF_K,
        )

        assert DEFAULT_CANDIDATE_DEPTH == 20
        assert DEFAULT_RRF_K == 60
        assert DEFAULT_FINAL_TOP_K == 5

    def test_oracle_citation_recall_tracks_citations_not_retrieval(self) -> None:
        """Section 18: oracle citation recall must respond to citing behaviour."""

        class CitingGen:
            def generate(self, prompt, images):
                return "The answer is grounded [T1]."

        class SilentGen:
            def generate(self, prompt, images):
                return "The answer is grounded, with no labels."

        query = self._query()
        recalls = {}
        for name, gen in (("cites", CitingGen()), ("silent", SilentGen())):
            runner = AblationRunner(self._stores(), generator=gen)
            result = runner.run_configuration(
                CONFIGURATIONS_BY_ID["oracle_evidence_generation"], query, {}, {}, generate=True
            )
            recalls[name] = result.generation_metrics["citation_recall"]
            # evidence availability is unchanged in both cases
            assert result.retrieval_metrics["recall_at_5"] == 1.0

        assert recalls["cites"] == 1.0
        assert recalls["silent"] == 0.0


class TestEvidenceHashIntegrityGrouping:
    """The integrity check must group by (query, configuration).

    Grouping by query alone conflates configurations that legitimately supply
    different evidence — most obviously the oracle, which supplies ground truth
    by design — and reports a false FAIL on a valid experiment.
    """

    def _row(self, query_id, config_id, condition, digest):
        row = QueryResult(query_id=query_id, config_id=config_id)
        row.generation_condition = condition
        row.evidence_context_hash = digest
        return row

    def _integrity(self, rows):
        import importlib.util
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[2] / "scripts/run_ablation.py"
        spec = importlib.util.spec_from_file_location("run_ablation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._evidence_hash_integrity(rows)

    def test_oracle_differing_from_retrieval_is_not_a_failure(self) -> None:
        rows = [
            self._row("q1", "full_reranked", "g0_baseline", "aaa"),
            self._row("q1", "full_reranked", "g3_structured_generation", "aaa"),
            # oracle supplies ground-truth evidence: a different hash by design
            self._row("q1", "oracle_evidence_generation", "g0_baseline", "bbb"),
            self._row("q1", "oracle_evidence_generation", "g3_structured_generation", "bbb"),
        ]
        result = self._integrity(rows)
        assert result["all_identical"] is True
        assert result["pairs_checked"] == 2

    def test_real_divergence_within_a_configuration_is_a_failure(self) -> None:
        rows = [
            self._row("q1", "full_reranked", "g0_baseline", "aaa"),
            self._row("q1", "full_reranked", "g3_structured_generation", "DIFFERENT"),
        ]
        result = self._integrity(rows)
        assert result["all_identical"] is False
        assert result["mismatched"] == [{"query_id": "q1", "config_id": "full_reranked"}]

    def test_per_configuration_counts_reported(self) -> None:
        rows = [
            self._row("q1", "full_reranked", "g0_baseline", "aaa"),
            self._row("q1", "full_reranked", "g1_explicit_citations", "aaa"),
            self._row("q2", "full_reranked", "g0_baseline", "ccc"),
            self._row("q2", "full_reranked", "g1_explicit_citations", "DIFFERENT"),
        ]
        result = self._integrity(rows)
        assert result["per_configuration"]["full_reranked"] == {"identical": 1, "divergent": 1}


class TestPairedAnalysisConfigurationSafety:
    """Paired comparison must never span configurations.

    A configuration determines what evidence the generator saw. Pairing a
    retrieval row against an oracle row compares two different contexts and
    silently answers a question nobody asked.
    """

    def _row(self, query_id, config_id, condition, f1):
        row = QueryResult(query_id=query_id, config_id=config_id)
        row.generation_condition = condition
        row.generation_metrics = {"citation_f1": f1, "citation_recall": f1}
        return row

    def _paired(self, rows):
        import importlib.util
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[2] / "scripts/run_ablation.py"
        spec = importlib.util.spec_from_file_location("run_ablation_paired", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._paired_analysis(rows)

    def test_mixed_configurations_are_rejected(self) -> None:
        rows = [
            self._row("q1", "full_reranked", "g0_baseline", 0.2),
            self._row("q1", "oracle_evidence_generation", "g0_baseline", 0.9),
        ]
        with pytest.raises(ValueError, match="one configuration"):
            self._paired(rows)

    def test_single_configuration_pairs_correctly(self) -> None:
        rows = [
            self._row("q1", "full_reranked", "g0_baseline", 0.2),
            self._row("q1", "full_reranked", "g3_structured_generation", 0.6),
            self._row("q2", "full_reranked", "g0_baseline", 0.5),
            self._row("q2", "full_reranked", "g3_structured_generation", 0.1),
        ]
        result = self._paired(rows)
        counts = result["g3_structured_generation_vs_g0_baseline"]["citation_f1"]
        assert counts["treatment_better"] == 1
        assert counts["baseline_better"] == 1
        assert counts["compared"] == 2
