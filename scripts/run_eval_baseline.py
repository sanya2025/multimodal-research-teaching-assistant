"""Baseline evaluation driver for PR1 — text-only VectorStore retrieval.

Loads a pinned evaluation corpus (data/eval/) and runs all benchmark queries
against the existing MRTA text retrieval system, computing Recall@5, Hit@5,
MRR, and nDCG@5, writing results to results/ (v1) or results/v2/ (v2).

Usage:
    python scripts/run_eval_baseline.py                  # v1 (default, unchanged)
    python scripts/run_eval_baseline.py --benchmark v2    # v2 (5 papers, 100 queries)

Requirements:
    - Ollama running with nomic-embed-text available (used for query embedding)
    - The benchmark's text vector store already built:
        v1: data/vector_store/aiayn/           (pre-built)
        v2: data/vector_store/v2_corpus/  (build with build_text_index.py --benchmark v2)

If Ollama is not available, the script exits with a clear error rather than
producing fabricated results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

K = 5

# Benchmark registry — v1 paths are byte-identical to the pre-generalization
# script; v2 adds a second, independent set of paths. Nothing about v1's
# behavior changes when --benchmark is omitted (default "v1").
BENCHMARKS = {
    "v1": {
        "vector_store": REPO_ROOT / "data" / "vector_store" / "aiayn",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json",
        "queries": REPO_ROOT / "data" / "eval" / "queries_v1.json",
        "results_dir": REPO_ROOT / "results",
        "metrics_filename": "baseline_metrics.json",
        "per_query_filename": "baseline_per_query.json",
    },
    "v2": {
        "vector_store": REPO_ROOT / "data" / "vector_store" / "v2_corpus",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json",
        "queries": REPO_ROOT / "data" / "eval" / "queries_v2.json",
        "results_dir": REPO_ROOT / "results" / "v2",
        "metrics_filename": "pr1_baseline_metrics.json",
        "per_query_filename": "pr1_per_query.json",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_list(query: dict) -> list[dict]:
    """v1 uses 'target_evidence'; v2 uses 'expected_evidence'. Support both."""
    return query.get("target_evidence", query.get("expected_evidence", []))


def _validate_dataset(dataset: dict) -> None:
    """Validate query count and intent distribution against the file's own
    declared counts — this generalizes across v1's 3 intents and v2's 5
    without hardcoding either taxonomy into the script."""
    queries = dataset.get("queries", [])
    expected_count = dataset.get("query_count", len(queries))
    if len(queries) != expected_count:
        raise ValueError(f"Expected {expected_count} queries, got {len(queries)}")

    counts: dict[str, int] = {}
    for q in queries:
        if "query_id" not in q or "intent" not in q or not _evidence_list(q):
            raise ValueError(f"Malformed or empty-evidence query: {q.get('query_id')}")
        counts[q["intent"]] = counts.get(q["intent"], 0) + 1

    expected_counts = dataset.get("counts_by_intent")
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Intent counts {counts} != declared {expected_counts}")

    ids = [q["query_id"] for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query_id values")


def _parse_targets(target_list: list[dict]) -> list:
    from mrta.eval.types import CanonicalEvidence

    return [
        CanonicalEvidence(
            document_id=t["document_id"],
            page_number=t["page_number"],
            figure_id=t.get("figure_id"),
        )
        for t in target_list
    ]


def _build_store(vector_store_path: Path) -> object:
    from mrta.retrieval.embedder import Embedder
    from mrta.retrieval.vector_store import VectorStore

    if not vector_store_path.exists():
        print(f"ERROR: vector store not found at {vector_store_path}")
        print("Build it first, e.g.: python scripts/build_text_index.py --benchmark v2")
        sys.exit(1)

    print(f"Loading FAISS index from {vector_store_path} ...")
    embedder = Embedder("nomic-embed-text")
    try:
        store = VectorStore.load(vector_store_path, embedder)
    except Exception as e:
        print(f"ERROR loading vector store: {e}")
        sys.exit(1)
    print(f"  {len(store._chunks)} chunks loaded")
    return store


def _probe_ollama() -> None:
    """Fail fast if Ollama is not reachable."""
    import httpx

    from mrta.core.config import settings

    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {settings.ollama_host}: {e}")
        print("Start Ollama with: ollama serve")
        sys.exit(1)


def _aggregate(per_query: list[dict], intent_filter: str | None = None) -> dict:
    if intent_filter is None:
        subset = per_query
    else:
        subset = [q for q in per_query if q["intent"] == intent_filter]
    if not subset:
        return {
            "sample_count": 0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
            "hit_at_5": 0.0,
        }
    n = len(subset)
    return {
        "sample_count": n,
        "recall_at_5": round(sum(q["recall_at_5"] for q in subset) / n, 4),
        "mrr": round(sum(q["mrr"] for q in subset) / n, 4),
        "ndcg_at_5": round(sum(q["ndcg_at_5"] for q in subset) / n, 4),
        "hit_at_5": round(sum(q["hit_at_5"] for q in subset) / n, 4),
    }


def _row(label: str, m: dict) -> str:
    return (
        f"  {label:20s}  {m['recall_at_5']:8.4f}  {m['mrr']:6.4f}"
        f"  {m['ndcg_at_5']:7.4f}  {m['hit_at_5']:6.4f}  (n={m['sample_count']})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from mrta.eval.adapter import EvalAdapter
    from mrta.eval.retrieval_metrics import (
        hit_rate_at_k,
        mean_reciprocal_rank,
        ndcg_at_k,
        recall_at_k,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
        default="v1",
        help="Which benchmark version to evaluate (default: v1, unchanged behavior)",
    )
    args = parser.parse_args()
    cfg = BENCHMARKS[args.benchmark]

    print(f"=== PR1 Baseline Evaluation ({args.benchmark}) ===")
    print()

    _probe_ollama()
    store = _build_store(cfg["vector_store"])

    manifest = _load_json(cfg["manifest"])
    dataset = _load_json(cfg["queries"])
    _validate_dataset(dataset)

    adapter = EvalAdapter(manifest)
    queries = dataset["queries"]
    intents = sorted(dataset.get("counts_by_intent", {}).keys()) or sorted(
        {q["intent"] for q in queries}
    )

    print(f"Running {len(queries)} queries at k={K} ...")
    print()

    per_query_results: list[dict] = []

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        intent = q["intent"]
        targets = _parse_targets(_evidence_list(q))

        try:
            raw_results = store.search_with_scores(query_text, k=K)
        except Exception as e:
            print(f"  {qid}: ERROR during retrieval: {e}")
            sys.exit(1)

        candidates = [
            adapter.chunk_to_candidate(chunk, score, rank + 1)
            for rank, (chunk, score) in enumerate(raw_results)
        ]

        r5 = recall_at_k(candidates, targets, K)
        h5 = hit_rate_at_k(candidates, targets, K)
        mrr = mean_reciprocal_rank(candidates, targets)
        nd5 = ndcg_at_k(candidates, targets, K)

        hit_symbol = "✓" if h5 > 0 else "✗"
        print(
            f"  {qid} [{intent:16s}] {hit_symbol}  Recall={r5:.2f}  MRR={mrr:.2f}"
            f"  nDCG={nd5:.2f}  | {query_text[:50]!r}"
        )

        retrieved_ev = [
            {
                "document_id": c.evidence.document_id,
                "page_number": c.evidence.page_number,
                "figure_id": c.evidence.figure_id,
                "score": round(c.score, 4),
            }
            for c in candidates
        ]
        expected_ev = [
            {"document_id": t.document_id, "page_number": t.page_number, "figure_id": t.figure_id}
            for t in targets
        ]

        per_query_results.append(
            {
                "query_id": qid,
                "query": query_text,
                "intent": intent,
                "document_id": q.get("document_id"),
                "recall_at_5": round(r5, 4),
                "hit_at_5": round(h5, 4),
                "mrr": round(mrr, 4),
                "ndcg_at_5": round(nd5, 4),
                "retrieved_canonical_evidence": retrieved_ev,
                "expected_canonical_evidence": expected_ev,
            }
        )

    print()
    print("=== Aggregated results ===")

    overall = _aggregate(per_query_results)
    by_intent = {intent: _aggregate(per_query_results, intent) for intent in intents}

    header = f"  {'':20s}  {'Recall@5':>8}  {'MRR':>6}  {'nDCG@5':>7}  {'Hit@5':>6}"
    print(header)
    print(_row("Overall", overall))
    for intent in intents:
        print(_row(intent, by_intent[intent]))

    baseline_metrics = {
        "status": "measured",
        "benchmark": args.benchmark,
        "dataset_version": dataset["dataset_version"],
        "corpus_version": dataset["corpus_version"],
        "retrieval_system": "VectorStore (text-only FAISS, nomic-embed-text)",
        "k": K,
        "sample_count": len(per_query_results),
        "metrics": {
            "overall": {k: v for k, v in overall.items() if k != "sample_count"},
            **{intent: by_intent[intent] for intent in intents},
        },
    }

    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    metrics_path = cfg["results_dir"] / cfg["metrics_filename"]
    per_query_path = cfg["results_dir"] / cfg["per_query_filename"]

    metrics_path.write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query_results, indent=2), encoding="utf-8")

    print()
    print(f"Saved → {metrics_path}")
    print(f"Saved → {per_query_path}")


if __name__ == "__main__":
    main()
