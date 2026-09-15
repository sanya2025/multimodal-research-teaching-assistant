"""Tests for mrta.eval.regression_gate and scripts/check_eval_regression.py.

The gate's job is to fail when it cannot be trusted, so most of these tests are
about the failure paths rather than the happy one. A checker that silently
passed on a missing metric would be worse than no checker: it would look like
evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mrta.eval.regression_gate import (
    DEFAULT_GATES,
    Baseline,
    CurrentRun,
    MetricComparison,
    RegressionGateError,
    compare,
    render_console,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_eval_regression.py"
REAL_BASELINE = REPO_ROOT / "results" / "v2" / "baselines" / "retrieval_regression_v1.json"

CONFIG_ID = "full_reranked"
BASE_METRICS = {"recall_at_5": 0.605, "mrr_at_5": 0.515333, "figure_recall_at_5": 0.48}
# The three retrieval models the identity gate compares. Shared so baseline and
# current agree by construction, and a test that overrides one overrides only one.
MODELS = {
    "embedding_model": "nomic-embed-text",
    "clip_model": "openai/clip-vit-base-patch32",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


# ----------------------------------------------------------------------
# Fixtures — minimal documents in the two real schemas
# ----------------------------------------------------------------------


def _baseline_doc(metrics: dict | None = None, **overrides) -> dict:
    doc = {
        "baseline_version": "test",
        "benchmark": {"name": "v2", "dataset_version": "v2.0.0"},
        "configuration": {
            "configuration_id": CONFIG_ID,
            "candidate_depth": 20,
            "rrf_k": 60,
            "final_top_k": 5,
        },
        "models": dict(MODELS),
        "metrics": dict(BASE_METRICS if metrics is None else metrics),
    }
    doc.update(overrides)
    return doc


def _current_doc(metrics: dict | None = None, metadata: dict | None = None) -> dict:
    meta = {
        "benchmark": "v2",
        "dataset_version": "v2.0.0",
        "candidate_depth": 20,
        "rrf_k": 60,
        "final_top_k": 5,
        "models": dict(MODELS),
    }
    if metadata:
        # Merge the models sub-dict rather than replacing it, so overriding one
        # model does not silently delete the other two and raise unrelated
        # identity errors alongside the one under test.
        models_override = metadata.pop("models", None)
        meta.update(metadata)
        if models_override:
            meta["models"] = {**meta["models"], **models_override}
    return {
        "metadata": meta,
        "overall": {CONFIG_ID: dict(BASE_METRICS if metrics is None else metrics)},
    }


@pytest.fixture
def write(tmp_path: Path):
    def _write(name: str, payload) -> Path:
        path = tmp_path / name
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )
        return path

    return _write


def _run(baseline: Path, current: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _gate(baseline_doc: dict, current_doc: dict, write, gates=None):
    b = Baseline.load(write("baseline.json", baseline_doc))
    c = CurrentRun.load(write("current.json", current_doc), CONFIG_ID)
    return compare(b, c, gates)


# ----------------------------------------------------------------------
# 1-3, 13: passing cases and drop semantics
# ----------------------------------------------------------------------


class TestPassingComparisons:
    def test_unchanged_metrics_pass(self, write):
        result = _gate(_baseline_doc(), _current_doc(), write)
        assert result.passed
        assert result.regressions == []

    def test_improvements_pass(self, write):
        better = {k: v + 0.1 for k, v in BASE_METRICS.items()}
        result = _gate(_baseline_doc(), _current_doc(better), write)
        assert result.passed
        assert all(c.delta > 0 for c in result.comparisons)

    def test_improvement_has_zero_drop(self, write):
        better = {**BASE_METRICS, "recall_at_5": 0.705}
        result = _gate(_baseline_doc(), _current_doc(better), write)
        recall = next(c for c in result.comparisons if c.name == "recall_at_5")
        assert recall.drop == 0.0
        assert recall.delta == pytest.approx(0.1)

    def test_recall_exactly_at_tolerance_passes(self, write):
        # 0.605 - 0.02 = 0.585. Binary floating point makes this subtraction
        # 0.01999999999999999, which is exactly why the gate carries an epsilon.
        at_edge = {**BASE_METRICS, "recall_at_5": 0.585}
        result = _gate(_baseline_doc(), _current_doc(at_edge), write)
        assert result.passed

    def test_drop_is_absolute_not_relative(self, write):
        # A 3.3% relative fall, but only 0.02 absolute — passes a 0.02 gate.
        # If the gate were relative this would be a different verdict.
        current = {**BASE_METRICS, "recall_at_5": 0.585}
        result = _gate(_baseline_doc(), _current_doc(current), write)
        recall = next(c for c in result.comparisons if c.name == "recall_at_5")
        assert recall.drop == pytest.approx(0.02)
        assert result.passed

    def test_delta_is_signed_drop_is_clamped(self):
        c = MetricComparison(name="m", baseline=0.5, current=0.6, allowed_drop=0.02)
        assert c.delta == pytest.approx(0.1)
        assert c.drop == 0.0


# ----------------------------------------------------------------------
# 4-7: regressions
# ----------------------------------------------------------------------


class TestRegressions:
    def test_recall_just_beyond_tolerance_fails(self, write):
        result = _gate(
            _baseline_doc(), _current_doc({**BASE_METRICS, "recall_at_5": 0.5849}), write
        )
        assert not result.passed
        assert [c.name for c in result.regressions] == ["recall_at_5"]

    def test_spec_example_0605_to_0580_fails(self, write):
        """The worked example from the PR9 spec: drop 0.025 against a 0.02 gate."""
        result = _gate(_baseline_doc(), _current_doc({**BASE_METRICS, "recall_at_5": 0.580}), write)
        recall = next(c for c in result.comparisons if c.name == "recall_at_5")
        assert recall.drop == pytest.approx(0.025)
        assert not recall.passed

    def test_mrr_beyond_tolerance_fails(self, write):
        current = {**BASE_METRICS, "mrr_at_5": 0.48}  # drop 0.0353 > 0.02
        result = _gate(_baseline_doc(), _current_doc(current), write)
        assert [c.name for c in result.regressions] == ["mrr_at_5"]

    def test_figure_recall_beyond_tolerance_fails(self, write):
        current = {**BASE_METRICS, "figure_recall_at_5": 0.44}  # drop 0.04 > 0.03
        result = _gate(_baseline_doc(), _current_doc(current), write)
        assert [c.name for c in result.regressions] == ["figure_recall_at_5"]

    def test_figure_recall_tolerance_is_looser_than_recall(self, write):
        # A 0.025 drop fails Recall@5 but passes Figure Recall@5. The two gates
        # are genuinely different, not copied.
        current = {"recall_at_5": 0.58, "mrr_at_5": 0.515333, "figure_recall_at_5": 0.455}
        result = _gate(_baseline_doc(), _current_doc(current), write)
        assert [c.name for c in result.regressions] == ["recall_at_5"]

    def test_multiple_regressions_all_reported(self, write):
        current = {"recall_at_5": 0.50, "mrr_at_5": 0.40, "figure_recall_at_5": 0.30}
        result = _gate(_baseline_doc(), _current_doc(current), write)
        assert len(result.regressions) == 3
        assert {c.name for c in result.regressions} == set(DEFAULT_GATES)

    def test_custom_tolerance_overrides_default(self, write):
        current = {**BASE_METRICS, "recall_at_5": 0.600}  # drop 0.005
        gates = {**DEFAULT_GATES, "recall_at_5": 0.001}
        result = _gate(_baseline_doc(), _current_doc(current), write, gates)
        assert [c.name for c in result.regressions] == ["recall_at_5"]


# ----------------------------------------------------------------------
# 8-12: fail-closed on unusable input
# ----------------------------------------------------------------------


class TestFailsClosed:
    def test_missing_baseline_file(self, tmp_path):
        with pytest.raises(RegressionGateError, match="not found"):
            Baseline.load(tmp_path / "nope.json")

    def test_missing_current_file(self, tmp_path):
        with pytest.raises(RegressionGateError, match="not found"):
            CurrentRun.load(tmp_path / "nope.json", CONFIG_ID)

    def test_malformed_baseline_json(self, write):
        with pytest.raises(RegressionGateError, match="not valid JSON"):
            Baseline.load(write("baseline.json", "{not json"))

    def test_malformed_current_json(self, write):
        with pytest.raises(RegressionGateError, match="not valid JSON"):
            CurrentRun.load(write("current.json", "[[["), CONFIG_ID)

    def test_json_array_rejected(self, write):
        with pytest.raises(RegressionGateError, match="JSON object"):
            Baseline.load(write("baseline.json", [1, 2, 3]))

    def test_baseline_missing_required_key(self, write):
        doc = _baseline_doc()
        del doc["metrics"]
        with pytest.raises(RegressionGateError, match="missing required key"):
            Baseline.load(write("baseline.json", doc))

    def test_baseline_missing_configuration_id(self, write):
        doc = _baseline_doc()
        del doc["configuration"]["configuration_id"]
        with pytest.raises(RegressionGateError, match="configuration_id"):
            Baseline.load(write("baseline.json", doc))

    def test_metric_missing_from_baseline(self, write):
        doc = _baseline_doc({"recall_at_5": 0.605, "mrr_at_5": 0.5})
        with pytest.raises(RegressionGateError, match="baseline has no metric"):
            _gate(doc, _current_doc(), write)

    def test_metric_missing_from_current(self, write):
        current = _current_doc({"recall_at_5": 0.605, "mrr_at_5": 0.5})
        with pytest.raises(RegressionGateError, match="current run has no metric"):
            _gate(_baseline_doc(), current, write)

    def test_unknown_configuration_in_current(self, write):
        current = CurrentRun.load  # noqa: F841 - readability only
        path = write("current.json", _current_doc())
        with pytest.raises(RegressionGateError, match="no results for configuration"):
            CurrentRun.load(path, "some_other_config")

    def test_current_without_overall_section(self, write):
        with pytest.raises(RegressionGateError, match="no 'overall' section"):
            CurrentRun.load(write("current.json", {"metadata": {}}), CONFIG_ID)

    def test_current_without_metadata_section(self, write):
        doc = {"overall": {CONFIG_ID: dict(BASE_METRICS)}}
        with pytest.raises(RegressionGateError, match="no 'metadata' section"):
            CurrentRun.load(write("current.json", doc), CONFIG_ID)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_current_metric(self, write, bad):
        # json.dumps emits bare NaN/Infinity, which json.loads accepts — so a
        # non-finite value really can reach the gate from a real artifact.
        with pytest.raises(RegressionGateError, match="not finite"):
            _gate(_baseline_doc(), _current_doc({**BASE_METRICS, "recall_at_5": bad}), write)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_baseline_metric(self, write, bad):
        with pytest.raises(RegressionGateError, match="not finite"):
            _gate(_baseline_doc({**BASE_METRICS, "mrr_at_5": bad}), _current_doc(), write)

    @pytest.mark.parametrize("bad", ["0.6", None, [0.6], {"v": 0.6}, True])
    def test_non_numeric_metric(self, write, bad):
        with pytest.raises(RegressionGateError, match="not a number"):
            _gate(_baseline_doc(), _current_doc({**BASE_METRICS, "recall_at_5": bad}), write)

    def test_negative_tolerance_rejected(self, write):
        with pytest.raises(RegressionGateError, match="negative"):
            _gate(_baseline_doc(), _current_doc(), write, {"recall_at_5": -0.1})

    def test_empty_gate_set_rejected(self, write):
        with pytest.raises(RegressionGateError, match="no metrics to gate"):
            _gate(_baseline_doc(), _current_doc(), write, {})


# ----------------------------------------------------------------------
# 14-15: configuration identity
# ----------------------------------------------------------------------


class TestConfigurationIdentity:
    def test_configuration_id_mismatch_fails(self, write):
        b = Baseline.load(write("baseline.json", _baseline_doc()))
        doc = _current_doc()
        doc["overall"] = {"text_caption_rrf": doc["overall"][CONFIG_ID]}
        c = CurrentRun.load(write("current.json", doc), "text_caption_rrf")
        result = compare(b, c)
        assert not result.passed
        assert any("configuration" in e for e in result.identity_errors)

    def test_full_reranked_baseline_rejects_text_caption_rrf_current(self, write):
        """The exact scenario named in the spec: incomparable experiments."""
        b = Baseline.load(write("baseline.json", _baseline_doc()))
        doc = _current_doc({"recall_at_5": 0.9, "mrr_at_5": 0.9, "figure_recall_at_5": 0.9})
        doc["overall"] = {"text_caption_rrf": doc["overall"][CONFIG_ID]}
        c = CurrentRun.load(write("current.json", doc), "text_caption_rrf")
        result = compare(b, c)
        # Metrics are better, yet the gate still fails — identity outranks score.
        assert all(comp.passed for comp in result.comparisons)
        assert not result.passed

    def test_benchmark_version_mismatch_fails(self, write):
        result = _gate(_baseline_doc(), _current_doc(metadata={"dataset_version": "v3.0.0"}), write)
        assert not result.passed
        assert any("dataset version" in e for e in result.identity_errors)

    def test_benchmark_name_mismatch_fails(self, write):
        result = _gate(_baseline_doc(), _current_doc(metadata={"benchmark": "v1"}), write)
        assert any("benchmark" in e for e in result.identity_errors)

    @pytest.mark.parametrize(
        "field,value,label",
        [
            ("candidate_depth", 50, "candidate depth"),
            ("rrf_k", 10, "RRF k"),
            ("final_top_k", 10, "final top-k"),
        ],
    )
    def test_retrieval_parameter_mismatch_fails(self, write, field, value, label):
        result = _gate(_baseline_doc(), _current_doc(metadata={field: value}), write)
        assert any(label in e for e in result.identity_errors)

    def test_reranker_model_mismatch_fails(self, write):
        metadata = {"models": {"reranker_model": "cross-encoder/other-model"}}
        result = _gate(_baseline_doc(), _current_doc(metadata=metadata), write)
        assert result.identity_errors == [
            "reranker model: baseline 'cross-encoder/ms-marco-MiniLM-L-6-v2' "
            "vs current 'cross-encoder/other-model'"
        ]

    def test_embedding_model_mismatch_fails(self, write):
        """The failure this gate exists for.

        A 384-d MiniLM query against the 768-d v2 index is caught by FAISS at
        search time. A *different 768-d* embedder is caught by nothing else:
        every score changes and no dimension check fires. Only this gate sees it.
        """
        metadata = {"models": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}}
        result = _gate(_baseline_doc(), _current_doc(metadata=metadata), write)
        assert not result.passed
        assert any("embedding model" in e for e in result.identity_errors)

    def test_embedding_model_mismatch_fails_even_when_metrics_improve(self, write):
        better = {"recall_at_5": 0.9, "mrr_at_5": 0.9, "figure_recall_at_5": 0.9}
        metadata = {"models": {"embedding_model": "some-other-768d-model"}}
        result = _gate(_baseline_doc(), _current_doc(better, metadata=metadata), write)
        assert all(c.passed for c in result.comparisons)
        assert not result.passed

    def test_clip_model_mismatch_fails(self, write):
        metadata = {"models": {"clip_model": "openai/clip-vit-large-patch14"}}
        result = _gate(_baseline_doc(), _current_doc(metadata=metadata), write)
        assert any("CLIP model" in e for e in result.identity_errors)

    def test_field_absent_on_both_sides_is_not_a_mismatch(self, write):
        doc = _baseline_doc()
        del doc["configuration"]["rrf_k"]
        current = _current_doc()
        del current["metadata"]["rrf_k"]
        result = _gate(doc, current, write)
        assert result.passed

    def test_field_absent_on_one_side_is_a_mismatch(self, write):
        current = _current_doc()
        del current["metadata"]["rrf_k"]
        result = _gate(_baseline_doc(), current, write)
        assert any("RRF k" in e for e in result.identity_errors)


# ----------------------------------------------------------------------
# 16: the baseline is never written
# ----------------------------------------------------------------------


class TestBaselineImmutability:
    def test_gate_does_not_modify_baseline_file(self, write):
        baseline_path = write("baseline.json", _baseline_doc())
        current_path = write(
            "current.json",
            _current_doc({"recall_at_5": 0.1, "mrr_at_5": 0.1, "figure_recall_at_5": 0.1}),
        )
        before = baseline_path.read_bytes()
        before_mtime = baseline_path.stat().st_mtime_ns

        result = compare(Baseline.load(baseline_path), CurrentRun.load(current_path, CONFIG_ID))
        assert not result.passed  # a failing run is the one most tempted to "fix" the baseline

        assert baseline_path.read_bytes() == before
        assert baseline_path.stat().st_mtime_ns == before_mtime

    def test_cli_does_not_modify_baseline_file(self, write):
        baseline_path = write("baseline.json", _baseline_doc())
        current_path = write("current.json", _current_doc())
        digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        _run(baseline_path, current_path)
        assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == digest

    def test_baseline_is_not_compared_against_itself(self, write):
        """A baseline file is not a valid 'current' file.

        Self-comparison would always pass and would look exactly like a real
        green run, so the schemas are deliberately not interchangeable.
        """
        baseline_path = write("baseline.json", _baseline_doc())
        with pytest.raises(RegressionGateError, match="no 'overall' section"):
            CurrentRun.load(baseline_path, CONFIG_ID)


# ----------------------------------------------------------------------
# The committed baseline, and the CLI contract
# ----------------------------------------------------------------------


class TestCommittedBaseline:
    def test_committed_baseline_loads(self):
        baseline = Baseline.load(REAL_BASELINE)
        assert baseline.configuration_id == "full_reranked"
        assert baseline.version == "v1"

    def test_committed_baseline_matches_pr7_artifact(self):
        """Guards against the baseline silently drifting from its stated source."""
        baseline = Baseline.load(REAL_BASELINE)
        pr7 = json.loads(
            (REPO_ROOT / "results" / "v2" / "pr7" / "ablation_summary.json").read_text()
        )["overall"]["full_reranked"]
        for name, value in baseline.metrics.items():
            assert value == pr7[name], f"{name} drifted from results/v2/pr7"

    def test_committed_baseline_matches_pr5_on_gated_metrics(self):
        """PR5 and PR7 measured the same configuration; the baseline must match both."""
        baseline = Baseline.load(REAL_BASELINE)
        pr5 = json.loads((REPO_ROOT / "results" / "v2" / "pr5_reranker_metrics.json").read_text())[
            "overall"
        ]["rerank_text_caption_clip"]
        assert baseline.metrics["recall_at_5"] == pr5["recall_at_5"]
        assert baseline.metrics["figure_recall_at_5"] == pr5["figure_recall_at_5"]
        assert baseline.metrics["mrr_at_5"] == pytest.approx(pr5["mrr"], abs=1e-4)

    def test_every_gated_metric_is_present_in_the_baseline(self):
        baseline = Baseline.load(REAL_BASELINE)
        assert set(DEFAULT_GATES) <= set(baseline.metrics)

    def test_baseline_declares_every_identity_field(self):
        """A field absent from the baseline is a field the gate cannot check."""
        baseline = Baseline.load(REAL_BASELINE)
        missing = [k for k, v in baseline.identity.items() if v is None]
        assert not missing, f"baseline cannot be identity-checked on: {missing}"

    def test_baseline_pins_the_embedding_model_used_to_build_the_indices(self):
        """Ties the baseline to nomic-embed-text, not to the repo default."""
        baseline = Baseline.load(REAL_BASELINE)
        index_config = json.loads(
            (REPO_ROOT / "data/eval/indices/v2/caption_index/config.json").read_text()
        )
        assert baseline.identity["embedding model"] == index_config["model"] == "nomic-embed-text"


class TestCli:
    def test_exit_zero_when_unchanged(self, write):
        proc = _run(write("b.json", _baseline_doc()), write("c.json", _current_doc()))
        assert proc.returncode == 0, proc.stderr
        assert "PASS" in proc.stdout

    def test_exit_one_on_regression(self, write):
        current = _current_doc({**BASE_METRICS, "recall_at_5": 0.4})
        proc = _run(write("b.json", _baseline_doc()), write("c.json", current))
        assert proc.returncode == 1
        assert "recall_at_5" in proc.stderr

    def test_exit_two_on_missing_file(self, write, tmp_path):
        proc = _run(write("b.json", _baseline_doc()), tmp_path / "absent.json")
        assert proc.returncode == 2

    def test_exit_two_on_configuration_mismatch(self, write):
        doc = _current_doc(metadata={"dataset_version": "v9"})
        proc = _run(write("b.json", _baseline_doc()), write("c.json", doc))
        assert proc.returncode == 2

    def test_cli_tolerance_flag_is_honoured(self, write):
        current = _current_doc({**BASE_METRICS, "recall_at_5": 0.600})
        args = (write("b.json", _baseline_doc()), write("c.json", current))
        assert _run(*args).returncode == 0
        assert _run(*args, "--recall5-max-drop", "0.001").returncode == 1

    def test_step_summary_written(self, write, tmp_path):
        summary = tmp_path / "summary.md"
        proc = _run(
            write("b.json", _baseline_doc()),
            write("c.json", _current_doc()),
            "--step-summary",
            str(summary),
        )
        assert proc.returncode == 0
        text = summary.read_text()
        assert "| Metric | Baseline | Current | Delta | Allowed Drop | Status |" in text
        assert "recall_at_5" in text


class TestRendering:
    def test_markdown_table_has_a_row_per_metric(self, write):
        b = Baseline.load(write("baseline.json", _baseline_doc()))
        c = CurrentRun.load(write("current.json", _current_doc()), CONFIG_ID)
        text = render_markdown(compare(b, c), baseline=b, current=c)
        for name in DEFAULT_GATES:
            assert f"| {name} |" in text

    def test_markdown_states_what_passing_does_not_prove(self, write):
        b = Baseline.load(write("baseline.json", _baseline_doc()))
        c = CurrentRun.load(write("current.json", _current_doc()), CONFIG_ID)
        text = render_markdown(compare(b, c), baseline=b, current=c)
        assert "does not prove semantic answer correctness" in text

    def test_markdown_lists_identity_errors(self, write):
        b = Baseline.load(write("baseline.json", _baseline_doc()))
        c = CurrentRun.load(write("current.json", _current_doc(metadata={"rrf_k": 1})), CONFIG_ID)
        text = render_markdown(compare(b, c), baseline=b, current=c)
        assert "Configuration mismatch" in text

    def test_console_marks_failures(self, write):
        result = _gate(_baseline_doc(), _current_doc({**BASE_METRICS, "recall_at_5": 0.1}), write)
        assert "FAIL" in render_console(result)


class TestSmokeEvalScript:
    def test_smoke_eval_passes(self):
        """Tier 1's plumbing check must itself be green, or the gate is untested."""
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "smoke_eval.py")],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "PASS" in proc.stdout

    def test_smoke_eval_labels_itself_as_synthetic(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "smoke_eval.py")],
            capture_output=True,
            text=True,
        )
        assert "not a benchmark result" in proc.stdout
