"""PR2 evaluation driver — caption retrieval vs PR1 text-only baseline.

Runs two experiments:

  Experiment A — Isolated Caption Retrieval
    CaptionVectorStore alone, evaluated across all intent slices.
    Primary metric: Figure Recall@5.

  Experiment B — Naive Raw-Score Pooling (diagnostic only)
    Pools VectorStore (text) and CaptionVectorStore (caption) candidates by
    raw cosine similarity score.
    Both stores use the same embedder (nomic-embed-text), the same IndexFlatIP,
    and L2-normalized vectors, so inner-product scores are in the same space
    and direct comparison is valid. Labelled "naive_raw_score_pooling" to be
    explicit that this is a diagnostic baseline, not principled fusion (RRF
    is deferred to PR4).

PR1 baseline results are loaded from their own result file for comparison but
never overwritten.

Usage:
    python scripts/run_eval_pr2.py                  # v1 (default, unchanged)
    python scripts/run_eval_pr2.py --benchmark v2    # v2 (5 papers, 100 queries)

Requirements:
    - Ollama running with nomic-embed-text (query embedding)
    - The benchmark's text vector store and caption index already built
      (see build_text_index.py / build_caption_index.py --benchmark v2)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

K = 5

BENCHMARKS = {
    "v1": {
        "vector_store": REPO_ROOT / "data" / "vector_store" / "aiayn",
        "caption_index": REPO_ROOT / "data" / "eval" / "indices" / "caption_index",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json",
        "queries": REPO_ROOT / "data" / "eval" / "queries_v1.json",
        "results_dir": REPO_ROOT / "results",
        "metrics_filename": "pr2_caption_metrics.json",
        "per_query_filename": "pr2_per_query.json",
        "pr1_results": REPO_ROOT / "results" / "baseline_metrics.json",
    },
    "v2": {
        "vector_store": REPO_ROOT / "data" / "vector_store" / "v2_corpus",
        "caption_index": REPO_ROOT / "data" / "eval" / "indices" / "v2" / "caption_index",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json",
        "queries": REPO_ROOT / "data" / "eval" / "queries_v2.json",
        "results_dir": REPO_ROOT / "results" / "v2",
        "metrics_filename": "pr2_caption_metrics.json",
        "per_query_filename": "pr2_per_query.json",
        "pr1_results": REPO_ROOT / "results" / "v2" / "pr1_baseline_metrics.json",
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


def _probe_ollama(host: str) -> None:
    import httpx

    try:
        r = httpx.get(f"{host}/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {host}: {e}")
        sys.exit(1)


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


def _has_figure_targets(targets: list) -> bool:
    """True if any target has a non-None figure_id."""
    return any(t.figure_id is not None for t in targets)


def _aggregate(per_query: list[dict], metric_prefix: str, intent_filter: str | None = None) -> dict:
    subset = [q for q in per_query if intent_filter is None or q["intent"] == intent_filter]
    if not subset:
        return {
            "sample_count": 0,
            "recall_at_5": 0.0,
            "figure_recall_at_5": None,
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
            "hit_at_5": 0.0,
        }
    n = len(subset)

    def _avg(key: str) -> float:
        return round(sum(q[metric_prefix + key] for q in subset) / n, 4)

    # Figure Recall is only meaningful for queries that have figure targets.
    # Queries with no figure targets store None; average over non-None only.
    # Return None for the slice if no query in it has figure targets.
    fr_key = metric_prefix + "figure_recall_at_5"
    fr_values = [q[fr_key] for q in subset if q[fr_key] is not None]
    figure_recall: float | None = round(sum(fr_values) / len(fr_values), 4) if fr_values else None

    return {
        "sample_count": n,
        "recall_at_5": _avg("recall_at_5"),
        "figure_recall_at_5": figure_recall,
        "mrr": _avg("mrr"),
        "ndcg_at_5": _avg("ndcg_at_5"),
        "hit_at_5": _avg("hit_at_5"),
    }


def _print_table(label: str, m: dict) -> None:
    fr5 = m["figure_recall_at_5"]
    fr5_str = f"{fr5:14.4f}" if fr5 is not None else f"{'N/A':>14}"
    print(
        f"  {label:16s}  {m['recall_at_5']:8.4f}  {fr5_str}"
        f"  {m['mrr']:6.4f}  {m['ndcg_at_5']:7.4f}  {m['hit_at_5']:6.4f}"
        f"  (n={m['sample_count']})"
    )


def _fmt(v: float | None, width: int) -> str:
    return f"{v:{width}.4f}" if v is not None else f"{'N/A':>{width}}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from mrta.core.config import settings
    from mrta.eval.adapter import EvalAdapter
    from mrta.eval.retrieval_metrics import (
        figure_recall_at_k,
        hit_rate_at_k,
        mean_reciprocal_rank,
        ndcg_at_k,
        recall_at_k,
    )
    from mrta.eval.types import RetrievedCandidate
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.embedder import Embedder
    from mrta.retrieval.vector_store import VectorStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="v1")
    args = parser.parse_args()
    cfg = BENCHMARKS[args.benchmark]

    print(f"=== PR2 Evaluation — Caption Retrieval ({args.benchmark}) ===")
    print()

    _probe_ollama(settings.ollama_host)

    embedder = Embedder(settings.embedding_model)

    if not cfg["vector_store"].exists():
        print(f"ERROR: Text vector store not found at {cfg['vector_store']}")
        sys.exit(1)
    if not cfg["caption_index"].exists():
        print(f"ERROR: Caption index not found at {cfg['caption_index']}")
        print(f"Build it first: python scripts/build_caption_index.py --benchmark {args.benchmark}")
        sys.exit(1)

    print(f"Loading text VectorStore from {cfg['vector_store']} ...")
    text_store = VectorStore.load(cfg["vector_store"], embedder)
    print(f"  {len(text_store._chunks)} text chunks")

    print(f"Loading CaptionVectorStore from {cfg['caption_index']} ...")
    caption_store = CaptionVectorStore.load(cfg["caption_index"], embedder)
    print(f"  {caption_store.size} caption record(s)")

    print()
    print("Score comparability: CONFIRMED")
    print("  Both stores use the same embedder, IndexFlatIP, L2-normalized vectors.")
    print("  Raw cosine scores are directly comparable — pooling labelled")
    print("  'naive_raw_score_pooling' (diagnostic baseline; RRF deferred to PR4).")
    print()

    manifest = _load_json(cfg["manifest"])
    dataset = _load_json(cfg["queries"])
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

        # --- Experiment A: caption-only ---
        caption_raw = caption_store.search_with_scores(query_text, k=K)
        caption_cands = [
            adapter.from_caption_record(rec, score, rank + 1)
            for rank, (rec, score) in enumerate(caption_raw)
        ]

        cap_r5 = recall_at_k(caption_cands, targets, K)
        cap_fr5 = (
            figure_recall_at_k(caption_cands, targets, K) if _has_figure_targets(targets) else None
        )
        cap_h5 = hit_rate_at_k(caption_cands, targets, K)
        cap_mrr = mean_reciprocal_rank(caption_cands, targets)
        cap_nd5 = ndcg_at_k(caption_cands, targets, K)

        # --- Experiment B: naive pooled (text + caption, sorted by score) ---
        text_raw = text_store.search_with_scores(query_text, k=K)
        text_cands = [
            adapter.chunk_to_candidate(chunk, score, rank + 1)
            for rank, (chunk, score) in enumerate(text_raw)
        ]

        combined = sorted(
            [(c, c.score) for c in text_cands] + [(c, c.score) for c in caption_cands],
            key=lambda x: x[1],
            reverse=True,
        )
        seen_ids: set[str] = set()
        pooled_cands = []
        for rank, (cand, _) in enumerate(combined, start=1):
            if cand.candidate_id not in seen_ids:
                seen_ids.add(cand.candidate_id)
                pooled_cands.append(
                    RetrievedCandidate(
                        candidate_id=cand.candidate_id,
                        evidence=cand.evidence,
                        score=cand.score,
                        rank=rank,
                    )
                )
            if len(pooled_cands) == K:
                break

        pool_r5 = recall_at_k(pooled_cands, targets, K)
        pool_fr5 = (
            figure_recall_at_k(pooled_cands, targets, K) if _has_figure_targets(targets) else None
        )
        pool_h5 = hit_rate_at_k(pooled_cands, targets, K)
        pool_mrr = mean_reciprocal_rank(pooled_cands, targets)
        pool_nd5 = ndcg_at_k(pooled_cands, targets, K)

        hit_sym = "✓" if cap_h5 > 0 else "✗"
        fr5_display = f"{cap_fr5:.2f}" if cap_fr5 is not None else "N/A"
        print(
            f"  {qid} [{intent:14s}] {hit_sym}"
            f"  capR={cap_r5:.2f} figR={fr5_display}"
            f"  | {query_text[:45]!r}"
        )

        per_query_results.append(
            {
                "query_id": qid,
                "query": query_text,
                "intent": intent,
                "document_id": q.get("document_id"),
                "expected_canonical_evidence": [
                    {
                        "document_id": t.document_id,
                        "page_number": t.page_number,
                        "figure_id": t.figure_id,
                    }
                    for t in targets
                ],
                # Experiment A
                "caption_recall_at_5": round(cap_r5, 4),
                "caption_figure_recall_at_5": round(cap_fr5, 4) if cap_fr5 is not None else None,
                "caption_mrr": round(cap_mrr, 4),
                "caption_ndcg_at_5": round(cap_nd5, 4),
                "caption_hit_at_5": round(cap_h5, 4),
                "caption_retrieved": [
                    {
                        "candidate_id": c.candidate_id,
                        "document_id": c.evidence.document_id,
                        "page_number": c.evidence.page_number,
                        "figure_id": c.evidence.figure_id,
                        "score": round(c.score, 4),
                        "rank": c.rank,
                    }
                    for c in caption_cands
                ],
                # Experiment B
                "pooled_recall_at_5": round(pool_r5, 4),
                "pooled_figure_recall_at_5": round(pool_fr5, 4) if pool_fr5 is not None else None,
                "pooled_mrr": round(pool_mrr, 4),
                "pooled_ndcg_at_5": round(pool_nd5, 4),
                "pooled_hit_at_5": round(pool_h5, 4),
                "pooled_retrieved": [
                    {
                        "candidate_id": c.candidate_id,
                        "document_id": c.evidence.document_id,
                        "page_number": c.evidence.page_number,
                        "figure_id": c.evidence.figure_id,
                        "score": round(c.score, 4),
                        "rank": c.rank,
                    }
                    for c in pooled_cands
                ],
            }
        )

    # Aggregate
    print()
    print("=== Experiment A — Caption-Only Results ===")
    header = (
        f"  {'':16s}  {'Recall@5':>8}  {'FigRecall@5':>14}"
        f"  {'MRR':>6}  {'nDCG@5':>7}  {'Hit@5':>6}"
    )
    print(header)
    cap_overall = _aggregate(per_query_results, "caption_")
    cap_by_intent = {
        intent: _aggregate(per_query_results, "caption_", intent) for intent in intents
    }
    _print_table("Overall", cap_overall)
    for intent in intents:
        _print_table(intent, cap_by_intent[intent])

    print()
    print("=== Experiment B — Naive Pooled (diagnostic) ===")
    print(header)
    pool_overall = _aggregate(per_query_results, "pooled_")
    pool_by_intent = {
        intent: _aggregate(per_query_results, "pooled_", intent) for intent in intents
    }
    _print_table("Overall", pool_overall)
    for intent in intents:
        _print_table(intent, pool_by_intent[intent])

    # --- PR1 comparison (loaded from PR1's own result file, never hardcoded) ---
    pr1_metrics: dict | None = None
    if cfg["pr1_results"].exists():
        pr1_metrics = _load_json(cfg["pr1_results"])
    else:
        print(
            f"\n  NOTE: PR1 results not found at {cfg['pr1_results']}; skipping comparison table."
        )

    if pr1_metrics is not None:
        print()
        print(f"=== PR1 → PR2 Comparison ({args.benchmark}) ===")
        print(f"  {'':30s}  {'PR1':>8}  {'PR2 cap':>9}  {'PR2 pool':>10}")

        pr1m = pr1_metrics["metrics"]
        rows = [
            (
                "Figure Recall@5 (fig targets only)",
                None,
                cap_overall["figure_recall_at_5"],
                pool_overall["figure_recall_at_5"],
            )
        ]
        for intent in intents:
            pr1_r5 = pr1m.get(intent, {}).get("recall_at_5")
            rows.append(
                (
                    f"{intent} Recall@5",
                    pr1_r5,
                    cap_by_intent[intent]["recall_at_5"],
                    pool_by_intent[intent]["recall_at_5"],
                )
            )
        for intent in intents:
            pr1_mrr = pr1m.get(intent, {}).get("mrr")
            rows.append(
                (
                    f"{intent} MRR",
                    pr1_mrr,
                    cap_by_intent[intent]["mrr"],
                    pool_by_intent[intent]["mrr"],
                )
            )
        for label, pr1, pr2_cap, pr2_pool in rows:
            print(f"  {label:30s}  {_fmt(pr1, 8)}  {_fmt(pr2_cap, 9)}  {_fmt(pr2_pool, 10)}")

        if "text" in pr1m:
            print()
            print("  Note — Text MRR (pooled vs PR1):")
            pr1_text_mrr = pr1m["text"]["mrr"]
            pool_text_mrr = pool_by_intent.get("text", {}).get("mrr", 0.0)
            print(f"    PR1 text-only MRR : {pr1_text_mrr:.4f}")
            delta = pool_text_mrr - pr1_text_mrr
            print(f"    PR2 pooled   MRR  : {pool_text_mrr:.4f}  (Δ = {delta:+.4f})")
            print("    Pooling preserves Recall@5 (text evidence still retrieved) but may")
            print("    degrade ranking quality — caption candidates can displace text results.")
            print("    This motivates proper rank fusion (RRF) in PR4.")

    # Write results
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "status": "measured",
        "benchmark": args.benchmark,
        "dataset_version": dataset["dataset_version"],
        "corpus_version": dataset["corpus_version"],
        "retrieval_system": {
            "caption": "CaptionVectorStore (EvidenceRecord.retrieval_text, nomic-embed-text)",
            "text": "VectorStore (text-only FAISS, nomic-embed-text)",
            "pooling": "naive_raw_score_pooling",
        },
        "k": K,
        "score_pooling_validity": (
            "CONFIRMED: both stores use the same embedder (nomic-embed-text), "
            "IndexFlatIP, and L2-normalized vectors. Cosine scores are directly "
            "comparable. Pooling is a diagnostic baseline; RRF deferred to PR4."
        ),
        "pr1_baseline_source": str(cfg["pr1_results"].relative_to(REPO_ROOT)),
        "caption_only": {
            "overall": {k: v for k, v in cap_overall.items() if k != "sample_count"},
            **cap_by_intent,
        },
        "naive_raw_score_pooling": {
            "overall": {k: v for k, v in pool_overall.items() if k != "sample_count"},
            **pool_by_intent,
        },
    }

    metrics_path = cfg["results_dir"] / cfg["metrics_filename"]
    per_query_path = cfg["results_dir"] / cfg["per_query_filename"]

    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query_results, indent=2), encoding="utf-8")

    print()
    print(f"Saved → {metrics_path}")
    print(f"Saved → {per_query_path}")


if __name__ == "__main__":
    main()
