"""Compare a fresh evaluation against a frozen baseline and gate on regressions.

The gate answers one narrow question: have selected frozen retrieval metrics
dropped further than an agreed engineering tolerance on the v2 benchmark?

Two things it deliberately is not:

*Statistics.* The tolerances are absolute metric-point allowances chosen to sit
above run-to-run noise, not confidence intervals. A pass is not evidence of no
change; a fail is not evidence of a significant one.

*A quality judgement.* Only retrieval metrics are gated. PR8 measured a
citation precision/recall trade-off whose relationship to semantic answer
quality is unresolved, so gating citation metrics would freeze one side of an
open question into CI. See ADR-012.

Every failure mode fails closed. A missing file, an absent metric, a NaN, or a
configuration that does not match the baseline all exit non-zero, because the
alternative — treating "could not compare" as "nothing regressed" — turns the
gate into decoration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Absolute metric-point tolerances, not relative percentages.
# Figure Recall@5 is allowed more room: it is computed over the 21 canonical
# figures rather than all 100 queries, so one figure moving is worth ~0.013 —
# an order of magnitude coarser than a single query moving Recall@5.
DEFAULT_GATES: dict[str, float] = {
    "recall_at_5": 0.02,
    "mrr_at_5": 0.02,
    "figure_recall_at_5": 0.03,
}

# Fields that must agree before two runs are comparable at all.
# (baseline path, current-metadata path, human label)
IDENTITY_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("benchmark.name", "benchmark", "benchmark"),
    ("benchmark.dataset_version", "dataset_version", "dataset version"),
    ("configuration.candidate_depth", "candidate_depth", "candidate depth"),
    ("configuration.rrf_k", "rrf_k", "RRF k"),
    ("configuration.final_top_k", "final_top_k", "final top-k"),
    # All three retrieval models, not just the reranker. A dimension change is
    # caught by FAISS at search time, but swapping in a *same-dimension* model —
    # another 768-d Ollama embedder, another 512-d CLIP — changes every score
    # while raising nothing. Without these the gate would compare two different
    # systems and call the difference a regression, or worse, a pass.
    ("models.embedding_model", "models.embedding_model", "embedding model"),
    ("models.clip_model", "models.clip_model", "CLIP model"),
    ("models.reranker_model", "models.reranker_model", "reranker model"),
)


class RegressionGateError(Exception):
    """Any condition that makes a trustworthy comparison impossible."""


# ----------------------------------------------------------------------
# Loading — every path fails closed
# ----------------------------------------------------------------------


def _dig(data: Any, dotted: str) -> Any:
    """Follow a dotted path, returning None if any step is missing."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def load_json(path: Path, *, label: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise RegressionGateError(f"{label} file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegressionGateError(f"{label} file is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise RegressionGateError(f"{label} file must contain a JSON object: {path}")
    return data


def _finite(value: Any, *, where: str) -> float:
    """Coerce to float, rejecting non-numerics and NaN/Infinity.

    bool is excluded explicitly: it is a subclass of int, and letting True
    become 1.0 would silently compare a flag against a metric.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegressionGateError(f"{where} is not a number: {value!r}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise RegressionGateError(f"{where} is not finite: {value!r}")
    return number


@dataclass(frozen=True)
class Baseline:
    """The frozen expectation. Read-only by construction and by discipline."""

    path: Path
    version: str
    configuration_id: str
    metrics: dict[str, float]
    identity: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Baseline:
        data = load_json(path, label="baseline")
        for key in ("baseline_version", "configuration", "metrics", "benchmark"):
            if key not in data:
                raise RegressionGateError(f"baseline is missing required key {key!r}: {path}")
        configuration_id = _dig(data, "configuration.configuration_id")
        if not configuration_id:
            raise RegressionGateError(f"baseline is missing configuration.configuration_id: {path}")
        metrics = data["metrics"]
        if not isinstance(metrics, dict) or not metrics:
            raise RegressionGateError(f"baseline has no metrics: {path}")
        return cls(
            path=Path(path),
            version=str(data["baseline_version"]),
            configuration_id=str(configuration_id),
            metrics={k: v for k, v in metrics.items()},
            identity={label: _dig(data, src) for src, _, label in IDENTITY_FIELDS},
        )


@dataclass(frozen=True)
class CurrentRun:
    """Metrics from the run under test, read from an ablation summary."""

    path: Path
    configuration_id: str
    metrics: dict[str, Any]
    identity: dict[str, Any]

    @classmethod
    def load(cls, path: Path, configuration_id: str) -> CurrentRun:
        data = load_json(path, label="current")
        overall = data.get("overall")
        if not isinstance(overall, dict):
            raise RegressionGateError(f"current file has no 'overall' section: {path}")
        if configuration_id not in overall:
            available = ", ".join(sorted(overall)) or "(none)"
            raise RegressionGateError(
                f"current file has no results for configuration {configuration_id!r}: {path} "
                f"(available: {available})"
            )
        metrics = overall[configuration_id]
        if not isinstance(metrics, dict):
            raise RegressionGateError(
                f"current results for {configuration_id!r} are not an object: {path}"
            )
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            raise RegressionGateError(f"current file has no 'metadata' section: {path}")
        return cls(
            path=Path(path),
            configuration_id=configuration_id,
            metrics=metrics,
            identity={label: _dig(metadata, src) for _, src, label in IDENTITY_FIELDS},
        )


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MetricComparison:
    name: str
    baseline: float
    current: float
    allowed_drop: float

    @property
    def delta(self) -> float:
        """Signed movement. Negative means the metric got worse."""
        return self.current - self.baseline

    @property
    def drop(self) -> float:
        """How far the metric fell. Zero when it improved or held."""
        return max(0.0, self.baseline - self.current)

    @property
    def passed(self) -> bool:
        # Tolerance is inclusive: a drop exactly at the allowance passes.
        # The epsilon absorbs float representation error only — 0.605 - 0.585
        # is 0.01999999999999999 in binary floating point, and failing that
        # would make the documented boundary a lie.
        return self.drop <= self.allowed_drop + 1e-9


@dataclass(frozen=True)
class GateResult:
    comparisons: list[MetricComparison]
    identity_errors: list[str]

    @property
    def regressions(self) -> list[MetricComparison]:
        return [c for c in self.comparisons if not c.passed]

    @property
    def passed(self) -> bool:
        return not self.identity_errors and not self.regressions


def check_identity(baseline: Baseline, current: CurrentRun) -> list[str]:
    """Report every way the two runs describe different experiments."""
    errors: list[str] = []
    if baseline.configuration_id != current.configuration_id:
        errors.append(
            f"configuration: baseline {baseline.configuration_id!r} "
            f"vs current {current.configuration_id!r}"
        )
    for label in (lbl for _, _, lbl in IDENTITY_FIELDS):
        want, got = baseline.identity.get(label), current.identity.get(label)
        # Absent on both sides is not a mismatch — older artifacts predate some
        # fields. Absent on one side is, because it cannot be verified.
        if want is None and got is None:
            continue
        if want != got:
            errors.append(f"{label}: baseline {want!r} vs current {got!r}")
    return errors


def compare(
    baseline: Baseline,
    current: CurrentRun,
    gates: dict[str, float] | None = None,
) -> GateResult:
    """Compare gated metrics, failing closed on anything unverifiable."""
    gates = DEFAULT_GATES if gates is None else gates
    if not gates:
        raise RegressionGateError("no metrics to gate on")

    comparisons: list[MetricComparison] = []
    for name, allowed in gates.items():
        allowed_drop = _finite(allowed, where=f"allowed drop for {name}")
        if allowed_drop < 0:
            raise RegressionGateError(f"allowed drop for {name} is negative: {allowed_drop}")
        if name not in baseline.metrics:
            raise RegressionGateError(f"baseline has no metric {name!r}: {baseline.path}")
        if name not in current.metrics:
            raise RegressionGateError(f"current run has no metric {name!r}: {current.path}")
        comparisons.append(
            MetricComparison(
                name=name,
                baseline=_finite(baseline.metrics[name], where=f"baseline {name}"),
                current=_finite(current.metrics[name], where=f"current {name}"),
                allowed_drop=allowed_drop,
            )
        )
    return GateResult(comparisons=comparisons, identity_errors=check_identity(baseline, current))


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def render_markdown(result: GateResult, *, baseline: Baseline, current: CurrentRun) -> str:
    """GitHub-flavoured summary table. Same content as the console output."""
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"## Evaluation regression gate — {status}",
        "",
        f"- Configuration: `{baseline.configuration_id}`",
        f"- Baseline: `{baseline.path}` (version {baseline.version})",
        f"- Current: `{current.path}`",
        "",
        "| Metric | Baseline | Current | Delta | Allowed Drop | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for c in result.comparisons:
        lines.append(
            f"| {c.name} | {c.baseline:.4f} | {c.current:.4f} | {c.delta:+.4f} "
            f"| {c.allowed_drop:.4f} | {'PASS' if c.passed else '**FAIL**'} |"
        )
    if result.identity_errors:
        lines += ["", "### Configuration mismatch", ""]
        lines += [f"- {e}" for e in result.identity_errors]
    lines += [
        "",
        "> Passing means selected frozen retrieval metrics have not regressed beyond "
        "configured tolerances on the v2 benchmark. It does not prove semantic answer "
        "correctness, generation faithfulness, or generalization beyond the benchmark.",
    ]
    return "\n".join(lines) + "\n"


def render_console(result: GateResult) -> str:
    header = (
        f"{'metric':<22}{'baseline':>10}{'current':>10}"
        f"{'delta':>10}{'drop':>9}{'allowed':>9}  result"
    )
    lines = [header, "-" * len(header)]
    for c in result.comparisons:
        lines.append(
            f"{c.name:<22}{c.baseline:>10.4f}{c.current:>10.4f}{c.delta:>+10.4f}"
            f"{c.drop:>9.4f}{c.allowed_drop:>9.4f}  {'PASS' if c.passed else 'FAIL'}"
        )
    if result.identity_errors:
        lines += ["", "configuration mismatch:"]
        lines += [f"  - {e}" for e in result.identity_errors]
    return "\n".join(lines)
