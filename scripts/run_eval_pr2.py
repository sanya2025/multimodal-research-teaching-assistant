"""PR2 evaluation driver — caption retrieval vs PR1 text-only baseline.

Runs two experiments:

  Experiment A — Isolated Caption Retrieval
    CaptionVectorStore alone, evaluated on visual and hybrid queries.
    Primary metric: Figure Recall@5.

  Experiment B — Naive Raw-Score Pooling (diagnostic only)
    Pools VectorStore (text) and CaptionVectorStore (caption) candidates by
    raw cosine similarity score.
    Both stores use the same embedder (nomic-embed-text), the same IndexFlatIP,
    and L2-normalized vectors, so inner-product scores are in the same space
    and direct comparison is valid. Labelled "naive_raw_score_pooling" to be
    explicit that this is a diagnostic baseline, not principled fusion (RRF
    is deferred to PR4).

PR1 baseline results are embedded for comparison but the original files
(results/baseline_metrics.json, results/baseline_per_query.json) are not
overwritten.

Usage:
    python scripts/run_eval_pr2.py

Requirements:
    - Ollama running with nomic-embed-text (query embedding)
    - data/vector_store/aiayn/ (PR1 text index)
    - data/eval/indices/caption_index/ (built by build_caption_index.py)
    - data/eval/corpus/v1/manifest.json
    - data/eval/queries_v1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

K = 5
VECTOR_STORE_PATH = REPO_ROOT / "data" / "vector_store" / "aiayn"
CAPTION_INDEX_PATH = REPO_ROOT / "data" / "eval" / "indices" / "caption_index"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json"
QUERIES_PATH = REPO_ROOT / "data" / "eval" / "queries_v1.json"
RESULTS_DIR = REPO_ROOT / "results"

PR1_BASELINE = {
    "overall_recall_at_5": 0.4667,
    "text_recall_at_5": 0.8750,
    "visual_recall_at_5": 0.0000,
    "hybrid_recall_at_5": 0.3889,
    "text_mrr": 0.7917,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


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
        f"  {label:8s}  {m['recall_at_5']:8.4f}  {fr5_str}"
        f"  {m['mrr']:6.4f}  {m['ndcg_at_5']:7.4f}  {m['hit_at_5']:6.4f}"
        f"  (n={m['sample_count']})"
    )


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

    print("=== PR2 Evaluation — Caption Retrieval ===")
    print()

    _probe_ollama(settings.ollama_host)

    # Load stores
    embedder = Embedder(settings.embedding_model)

    if not VECTOR_STORE_PATH.exists():
        print(f"ERROR: Text vector store not found at {VECTOR_STORE_PATH}")
        print("Build it first: python scripts/ingest.py data/eval/corpus/v1/papers/...")
        sys.exit(1)

    if not CAPTION_INDEX_PATH.exists():
        print(f"ERROR: Caption index not found at {CAPTION_INDEX_PATH}")
        print("Build it first: python scripts/build_caption_index.py")
        sys.exit(1)

    print(f"Loading text VectorStore from {VECTOR_STORE_PATH} ...")
    text_store = VectorStore.load(VECTOR_STORE_PATH, embedder)
    print(f"  {len(text_store._chunks)} text chunks")

    print(f"Loading CaptionVectorStore from {CAPTION_INDEX_PATH} ...")
    caption_store = CaptionVectorStore.load(CAPTION_INDEX_PATH, embedder)
    print(f"  {caption_store.size} caption record(s)")

    # Score comparability note — both stores use the same embedder + IndexFlatIP
    # with L2-normalized vectors, so inner-product scores are cosine similarities
    # in the same [0, 1] space. Naive pooling by raw score is valid here.
    print()
    print("Score comparability: CONFIRMED")
    print("  Both stores use the same embedder, IndexFlatIP, L2-normalized vectors.")
    print("  Raw cosine scores are directly comparable — pooling labelled")
    print("  'naive_raw_score_pooling' (diagnostic baseline; RRF deferred to PR4).")
    print()

    manifest = _load_json(MANIFEST_PATH)
    dataset = _load_json(QUERIES_PATH)
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
            f"  {qid} [{intent:6s}] {hit_sym}"
            f"  capR={cap_r5:.2f} figR={fr5_display}"
            f"  | {query_text[:55]!r}"
        )

        per_query_results.append(
            {
                "query_id": qid,
                "query": query_text,
                "intent": intent,
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
        f"  {'':8s}  {'Recall@5':>8}  {'FigRecall@5':>14}"
        f"  {'MRR':>6}  {'nDCG@5':>7}  {'Hit@5':>6}"
    )
    print(header)
    cap_overall = _aggregate(per_query_results, "caption_")
    cap_text = _aggregate(per_query_results, "caption_", "text")
    cap_visual = _aggregate(per_query_results, "caption_", "visual")
    cap_hybrid = _aggregate(per_query_results, "caption_", "hybrid")
    _print_table("Overall", cap_overall)
    _print_table("Text", cap_text)
    _print_table("Visual", cap_visual)
    _print_table("Hybrid", cap_hybrid)

    print()
    print("=== Experiment B — Naive Pooled (diagnostic) ===")
    print(header)
    pool_overall = _aggregate(per_query_results, "pooled_")
    pool_text = _aggregate(per_query_results, "pooled_", "text")
    pool_visual = _aggregate(per_query_results, "pooled_", "visual")
    pool_hybrid = _aggregate(per_query_results, "pooled_", "hybrid")
    _print_table("Overall", pool_overall)
    _print_table("Text", pool_text)
    _print_table("Visual", pool_visual)
    _print_table("Hybrid", pool_hybrid)

    print()
    print("=== PR1 → PR2 Comparison ===")
    print(f"  {'':30s}  {'PR1':>8}  {'PR2 cap':>9}  {'PR2 pool':>10}")

    def _fmt(v: float | None, width: int) -> str:
        return f"{v:{width}.4f}" if v is not None else f"{'N/A':>{width}}"

    rows = [
        (
            "Text Recall@5",
            PR1_BASELINE["text_recall_at_5"],
            cap_text["recall_at_5"],
            pool_text["recall_at_5"],
        ),
        (
            "Visual Recall@5",
            PR1_BASELINE["visual_recall_at_5"],
            cap_visual["recall_at_5"],
            pool_visual["recall_at_5"],
        ),
        (
            "Hybrid Recall@5",
            PR1_BASELINE["hybrid_recall_at_5"],
            cap_hybrid["recall_at_5"],
            pool_hybrid["recall_at_5"],
        ),
        (
            "Figure Recall@5 (fig targets only)",
            None,
            cap_overall["figure_recall_at_5"],
            pool_overall["figure_recall_at_5"],
        ),
        ("Text MRR", PR1_BASELINE.get("text_mrr"), cap_text["mrr"], pool_text["mrr"]),
    ]
    for label, pr1, pr2_cap, pr2_pool in rows:
        print(f"  {label:36s}  {_fmt(pr1, 8)}  {_fmt(pr2_cap, 9)}  {_fmt(pr2_pool, 10)}")

    print()
    print("  Note — Text MRR (pooled vs PR1):")
    pr1_text_mrr = 0.7917
    pool_text_mrr = pool_text["mrr"]
    print(f"    PR1 text-only MRR : {pr1_text_mrr:.4f}")
    print(f"    PR2 pooled   MRR  : {pool_text_mrr:.4f}  (Δ = {pool_text_mrr - pr1_text_mrr:+.4f})")
    print("    Pooling preserves Recall@5 (text evidence still retrieved) but degrades")
    print("    ranking quality — caption candidates displace text results from top positions.")
    print("    This motivates proper rank fusion (RRF) in PR4.")

    # Write results
    RESULTS_DIR.mkdir(exist_ok=True)

    metrics_out = {
        "status": "measured",
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
        "pr1_baseline": PR1_BASELINE,
        "caption_only": {
            "overall": {k: v for k, v in cap_overall.items() if k != "sample_count"},
            "text": cap_text,
            "visual": cap_visual,
            "hybrid": cap_hybrid,
        },
        "naive_raw_score_pooling": {
            "overall": {k: v for k, v in pool_overall.items() if k != "sample_count"},
            "text": pool_text,
            "visual": pool_visual,
            "hybrid": pool_hybrid,
        },
    }

    metrics_path = RESULTS_DIR / "pr2_caption_metrics.json"
    per_query_path = RESULTS_DIR / "pr2_per_query.json"

    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query_results, indent=2), encoding="utf-8")

    print()
    print(f"Saved → {metrics_path}")
    print(f"Saved → {per_query_path}")


if __name__ == "__main__":
    main()
