"""Gate a fresh evaluation against the frozen retrieval baseline.

Usage:
    python scripts/check_eval_regression.py \
        --baseline results/v2/baselines/retrieval_regression_v1.json \
        --current artifacts/eval/ablation_summary.json

    # tighten or relax a tolerance explicitly (absolute metric points)
    python scripts/check_eval_regression.py ... --recall5-max-drop 0.01

Exit codes:
    0  every gated metric is within tolerance
    1  a metric regressed beyond tolerance
    2  the comparison could not be made (missing/invalid input, config mismatch)

Independent of GitHub Actions: --step-summary is optional, and the same table
is always printed to stdout.

The tolerances are absolute metric-point allowances, not relative percentages
and not statistical bounds. A baseline of 0.605 against a current of 0.580 is a
drop of 0.025 and fails a 0.02 gate.

Never regenerate the baseline to make this pass. See ADR-012.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_BASELINE = REPO_ROOT / "results" / "v2" / "baselines" / "retrieval_regression_v1.json"

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--configuration",
        default=None,
        help="Configuration id to read from the current summary. "
        "Defaults to the baseline's own configuration_id.",
    )
    # Absolute drops. Defaults live in DEFAULT_GATES so script and library agree.
    parser.add_argument("--recall5-max-drop", type=float, default=None)
    parser.add_argument("--mrr5-max-drop", type=float, default=None)
    parser.add_argument("--figure-recall5-max-drop", type=float, default=None)
    parser.add_argument(
        "--step-summary",
        type=Path,
        default=None,
        help="Write a markdown table here. Defaults to $GITHUB_STEP_SUMMARY when set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from mrta.eval.regression_gate import (
        DEFAULT_GATES,
        Baseline,
        CurrentRun,
        RegressionGateError,
        compare,
        render_console,
        render_markdown,
    )

    args = build_parser().parse_args(argv)

    gates = dict(DEFAULT_GATES)
    for flag, metric in (
        (args.recall5_max_drop, "recall_at_5"),
        (args.mrr5_max_drop, "mrr_at_5"),
        (args.figure_recall5_max_drop, "figure_recall_at_5"),
    ):
        if flag is not None:
            gates[metric] = flag

    try:
        baseline = Baseline.load(args.baseline)
        current = CurrentRun.load(args.current, args.configuration or baseline.configuration_id)
        result = compare(baseline, current, gates)
    except RegressionGateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID

    print(f"baseline:      {baseline.path} (version {baseline.version})")
    print(f"current:       {current.path}")
    print(f"configuration: {baseline.configuration_id}")
    print()
    print(render_console(result))
    print()

    summary_path = args.step_summary or (
        Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None
    )
    if summary_path is not None:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(render_markdown(result, baseline=baseline, current=current))

    if result.identity_errors:
        print("FAIL: baseline and current describe different experiments.", file=sys.stderr)
        return EXIT_INVALID
    if result.regressions:
        names = ", ".join(c.name for c in result.regressions)
        print(f"FAIL: {len(result.regressions)} metric(s) regressed: {names}", file=sys.stderr)
        return EXIT_REGRESSION

    print("PASS: no gated metric regressed beyond tolerance.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
