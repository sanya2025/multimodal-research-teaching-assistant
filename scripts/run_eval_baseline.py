"""Baseline evaluation driver for PR1 — text-only VectorStore retrieval.

Loads the pinned evaluation corpus (data/eval/), runs all 20 benchmark queries
against the existing MRTA text retrieval system, computes Recall@5, Hit@5,
MRR, and nDCG@5, and writes results to results/.

Usage:
    python scripts/run_eval_baseline.py

Requirements:
    - Ollama running with nomic-embed-text available (used for query embedding)
    - data/vector_store/aiayn/ exists (pre-built FAISS index for the corpus)

If Ollama is not available, the script exits with a clear error rather than
producing fabricated results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

K = 5
VECTOR_STORE_PATH = REPO_ROOT / "data" / "vector_store" / "aiayn"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json"
QUERIES_PATH = REPO_ROOT / "data" / "eval" / "queries_v1.json"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_dataset(dataset: dict) -> None:
    queries = dataset.get("queries", [])
    if len(queries) != 20:
        raise ValueError(f"Expected 20 queries, got {len(queries)}")
    counts: dict[str, int] = {}
    for q in queries:
        if "query_id" not in q or "intent" not in q or "target_evidence" not in q:
            raise ValueError(f"Malformed query: {q.get('query_id')}")
        if not q["target_evidence"]:
            raise ValueError(f"Empty target_evidence for {q['query_id']}")
        counts[q["intent"]] = counts.get(q["intent"], 0) + 1
    expected = {"text": 8, "visual": 6, "hybrid": 6}
    if counts != expected:
        raise ValueError(f"Intent counts {counts} != expected {expected}")
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


def _build_store() -> object:
    from mrta.retrieval.embedder import Embedder
    from mrta.retrieval.vector_store import VectorStore

    if not VECTOR_STORE_PATH.exists():
        print(f"ERROR: vector store not found at {VECTOR_STORE_PATH}")
        print("Build it first: python scripts/ingest.py "
              "data/eval/corpus/v1/papers/attention_is_all_you_need.pdf")
        sys.exit(1)

    print("Loading FAISS index from data/vector_store/aiayn/ ...")
    embedder = Embedder("nomic-embed-text")
    try:
        store = VectorStore.load(VECTOR_STORE_PATH, embedder)
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
        return {"sample_count": 0, "recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0,
                "hit_at_5": 0.0}
    n = len(subset)
    return {
        "sample_count": n,
        "recall_at_5": round(sum(q["recall_at_5"] for q in subset) / n, 4),
        "mrr": round(sum(q["mrr"] for q in subset) / n, 4),
        "ndcg_at_5": round(sum(q["ndcg_at_5"] for q in subset) / n, 4),
        "hit_at_5": round(sum(q["hit_at_5"] for q in subset) / n, 4),
    }


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

    print("=== PR1 Baseline Evaluation ===")
    print()

    _probe_ollama()
    store = _build_store()

    manifest = _load_json(MANIFEST_PATH)
    dataset = _load_json(QUERIES_PATH)
    _validate_dataset(dataset)

    adapter = EvalAdapter(manifest)
    queries = dataset["queries"]

    print(f"Running {len(queries)} queries at k={K} ...")
    print()

    per_query_results: list[dict] = []

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        intent = q["intent"]
        targets = _parse_targets(q["target_evidence"])

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
            f"  {qid} [{intent:6s}] {hit_symbol}  Recall={r5:.2f}  MRR={mrr:.2f}"
            f"  nDCG={nd5:.2f}  | {query_text[:60]!r}"
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

        per_query_results.append({
            "query_id": qid,
            "query": query_text,
            "intent": intent,
            "recall_at_5": round(r5, 4),
            "hit_at_5": round(h5, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_5": round(nd5, 4),
            "retrieved_canonical_evidence": retrieved_ev,
            "expected_canonical_evidence": expected_ev,
        })

    print()
    print("=== Aggregated results ===")

    overall = _aggregate(per_query_results)
    text_agg = _aggregate(per_query_results, "text")
    visual_agg = _aggregate(per_query_results, "visual")
    hybrid_agg = _aggregate(per_query_results, "hybrid")

    header = f"  {'':8s}  {'Recall@5':>8}  {'MRR':>6}  {'nDCG@5':>7}  {'Hit@5':>6}"
    print(header)
    def _row(label: str, m: dict) -> str:
        return (
            f"  {label:8s}  {m['recall_at_5']:8.4f}  {m['mrr']:6.4f}"
            f"  {m['ndcg_at_5']:7.4f}  {m['hit_at_5']:6.4f}  (n={m['sample_count']})"
        )

    print(_row("Overall", overall))
    print(_row("Text", text_agg))
    print(_row("Visual", visual_agg))
    print(_row("Hybrid", hybrid_agg))

    baseline_metrics = {
        "status": "measured",
        "dataset_version": dataset["dataset_version"],
        "corpus_version": dataset["corpus_version"],
        "retrieval_system": "VectorStore (text-only FAISS, nomic-embed-text)",
        "k": K,
        "sample_count": len(per_query_results),
        "metrics": {
            "overall": {k: v for k, v in overall.items() if k != "sample_count"},
            "text": text_agg,
            "visual": visual_agg,
            "hybrid": hybrid_agg,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "baseline_metrics.json"
    per_query_path = RESULTS_DIR / "baseline_per_query.json"

    metrics_path.write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query_results, indent=2), encoding="utf-8")

    print()
    print(f"Saved → {metrics_path}")
    print(f"Saved → {per_query_path}")


if __name__ == "__main__":
    main()
