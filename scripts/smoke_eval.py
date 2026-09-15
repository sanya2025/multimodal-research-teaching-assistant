"""Deterministic plumbing check for the evaluation path — Tier 1 CI.

Exercises the full chain on synthetic data:

    query -> fake retrieval -> evaluator -> metrics JSON -> regression checker

The point is that the *wiring* survives a refactor: the real metric functions,
the real aggregator, the real summary schema and the real gate all run. Nothing
here loads a model, reads the benchmark, or opens a socket, so it fits in a
pull-request job.

These numbers are NOT a benchmark result. They come from four hand-made
candidates chosen to make the arithmetic checkable by hand, and must never be
quoted as a measurement of the system. The real numbers come from Tier 2.

Usage:
    python scripts/smoke_eval.py
    python scripts/smoke_eval.py --keep-output-dir artifacts/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG_ID = "smoke_fake_retrieval"

# Two queries with hand-computable outcomes.
#
#   q1 target (docA, p2)  -> arrives at rank 2  => recall@5 1.0, RR 1/2
#   q2 target (docB, p7, fig_b1) and (docB, p9)
#                         -> only the figure is found, at rank 1
#                            => recall@5 0.5, RR 1/1, figure recall@5 1.0
#
# Averaged over the two queries:
#   recall_at_5        = (1.0 + 0.5) / 2 = 0.75
#   mrr_at_5           = (0.5 + 1.0) / 2 = 0.75
#   figure_recall_at_5 = 1.0  (only q2 has a figure target, so only q2 counts)
EXPECTED_METRICS = {
    "recall_at_5": 0.75,
    "mrr_at_5": 0.75,
    "figure_recall_at_5": 1.0,
}


def _fake_retrieval() -> list[dict]:
    """Two canned (query, ranked candidates, targets) triples. No model, no index."""
    return [
        {
            "query_id": "smoke_q1",
            "intent": "text",
            "document_id": "docA",
            "targets": [("docA", 2, None)],
            "candidates": [("docA", 5, None), ("docA", 2, None), ("docB", 1, None)],
        },
        {
            "query_id": "smoke_q2",
            "intent": "visual_caption",
            "document_id": "docB",
            "targets": [("docB", 7, "fig_b1"), ("docB", 9, None)],
            "candidates": [("docB", 7, "fig_b1"), ("docA", 3, None)],
        },
    ]


def run_smoke(output_dir: Path) -> tuple[dict, Path]:
    """Drive the real evaluator over fake retrieval; return (summary, summary_path)."""
    from mrta.eval.ablation import QueryResult, aggregate, compute_retrieval_metrics
    from mrta.eval.types import CanonicalEvidence, RetrievedCandidate

    rows: list[QueryResult] = []
    for case in _fake_retrieval():
        targets = [CanonicalEvidence(*t) for t in case["targets"]]
        candidates = [
            RetrievedCandidate(
                candidate_id=f"{case['query_id']}_c{i}",
                evidence=CanonicalEvidence(*c),
                score=1.0 - 0.1 * i,
                rank=i + 1,
            )
            for i, c in enumerate(case["candidates"])
        ]
        rows.append(
            QueryResult(
                query_id=case["query_id"],
                config_id=CONFIG_ID,
                document_id=case["document_id"],
                intent=case["intent"],
                expected_evidence=[t.__dict__ for t in targets],
                retrieval_metrics=compute_retrieval_metrics(candidates, targets),
            )
        )

    summary = {
        "metadata": {
            "benchmark": "smoke",
            "dataset_version": "smoke-1",
            "query_count": len(rows),
            "candidate_depth": 20,
            "rrf_k": 60,
            "final_top_k": 5,
            "models": {
                "embedding_model": "fake/none",
                "clip_model": None,
                "reranker_model": "fake/none",
                "generator_model": None,
            },
            "note": "Synthetic plumbing check. Not a benchmark measurement.",
        },
        "overall": {CONFIG_ID: aggregate(rows)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, summary_path


def write_smoke_baseline(output_dir: Path) -> Path:
    """A baseline matching the expected metrics, so the gate has something to compare."""
    baseline = {
        "baseline_version": "smoke",
        "benchmark": {"name": "smoke", "dataset_version": "smoke-1"},
        "configuration": {
            "configuration_id": CONFIG_ID,
            "candidate_depth": 20,
            "rrf_k": 60,
            "final_top_k": 5,
        },
        # Must mirror every model field the summary records: the identity gate
        # treats a field present on one side and absent on the other as a mismatch.
        "models": {
            "embedding_model": "fake/none",
            "clip_model": None,
            "reranker_model": "fake/none",
        },
        "metrics": dict(EXPECTED_METRICS),
    }
    path = output_dir / "smoke_baseline.json"
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-output-dir",
        type=Path,
        default=None,
        help="Write artifacts here instead of a temporary directory.",
    )
    args = parser.parse_args(argv)

    from mrta.eval.regression_gate import (
        DEFAULT_GATES,
        Baseline,
        CurrentRun,
        compare,
        render_console,
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = args.keep_output_dir or Path(tmp)
        summary, summary_path = run_smoke(output_dir)
        metrics = summary["overall"][CONFIG_ID]

        print("=== smoke evaluation (synthetic — not a benchmark result) ===")
        failures = []
        for name, expected in EXPECTED_METRICS.items():
            actual = metrics.get(name)
            ok = actual is not None and abs(actual - expected) < 1e-9
            print(f"  {name:<20} expected {expected:<8} got {actual}  {'OK' if ok else 'MISMATCH'}")
            if not ok:
                failures.append(name)
        if failures:
            print(f"\nFAIL: evaluator produced unexpected metrics: {', '.join(failures)}")
            return 1

        baseline_path = write_smoke_baseline(output_dir)
        baseline = Baseline.load(baseline_path)
        current = CurrentRun.load(summary_path, CONFIG_ID)
        # The production tolerances, so the smoke run also exercises the real
        # gate configuration rather than a bespoke one.
        result = compare(baseline, current, dict(DEFAULT_GATES))

        print("\n=== regression checker ===")
        print(render_console(result))
        if not result.passed:
            print("\nFAIL: regression checker rejected its own reference values.")
            return 1

    print("\nPASS: evaluator and regression checker are wired correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
