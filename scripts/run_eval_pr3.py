"""PR3 evaluation driver — isolated CLIP visual retrieval, and caption-vs-CLIP comparison.

Experiment A — Isolated CLIP Retrieval
    ImageStore alone (openai/clip-vit-base-patch32), text query → image search.
    Reported over all intent slices; visual/hybrid slices are the meaningful ones.

Experiment B — Caption vs CLIP Complementarity
    For every query carrying figure targets, records where each stream ranked the
    correct figure, Hit@1 for each, and top-k figure-ID overlap. This answers
    whether the two streams behave differently, which is the evidence PR4 needs.

Deliberately NOT done here: raw scores from CLIP (512-D CLIP space) and from
nomic-embed-text retrieval are never pooled into one ranking. Those spaces are
uncalibrated with respect to each other, so sorting their cosines together would
be meaningless. Cross-stream combination is a rank-fusion concern (PR4).

Usage:
    python scripts/run_eval_pr3.py                  # v1 (default, unchanged)
    python scripts/run_eval_pr3.py --benchmark v2    # v2 (5 papers, 100 queries)

Requirements:
    - The benchmark's CLIP index (build_clip_image_index.py --benchmark ...)
    - The benchmark's caption index (build_caption_index.py --benchmark ..., for Experiment B)
    - Ollama running with nomic-embed-text (Experiment B query embedding only)

If Ollama is unavailable, Experiment A still runs and Experiment B is reported as
unavailable rather than estimated.
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
        "clip_index": REPO_ROOT / "data" / "eval" / "indices" / "clip_image_index",
        "caption_index": REPO_ROOT / "data" / "eval" / "indices" / "caption_index",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json",
        "queries": REPO_ROOT / "data" / "eval" / "queries_v1.json",
        "results_dir": REPO_ROOT / "results",
        "metrics_filename": "pr3_clip_metrics.json",
        "per_query_filename": "pr3_per_query.json",
        "pr1_results": REPO_ROOT / "results" / "baseline_metrics.json",
    },
    "v2": {
        "clip_index": REPO_ROOT / "data" / "eval" / "indices" / "v2" / "clip_image_index",
        "caption_index": REPO_ROOT / "data" / "eval" / "indices" / "v2" / "caption_index",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json",
        "queries": REPO_ROOT / "data" / "eval" / "queries_v2.json",
        "results_dir": REPO_ROOT / "results" / "v2",
        "metrics_filename": "pr3_clip_metrics.json",
        "per_query_filename": "pr3_per_query.json",
        "pr1_results": REPO_ROOT / "results" / "v2" / "pr1_baseline_metrics.json",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_list(query: dict) -> list[dict]:
    """v1 uses 'target_evidence'; v2 uses 'expected_evidence'. Support both."""
    return query.get("target_evidence", query.get("expected_evidence", []))


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
    """True if any target names a specific figure."""
    return any(t.figure_id is not None for t in targets)


def _figure_rank(candidates: list, targets: list) -> int | None:
    """1-based rank of the first candidate matching any figure target, else None."""
    figure_targets = [t for t in targets if t.figure_id is not None]
    for i, cand in enumerate(candidates, start=1):
        for target in figure_targets:
            if target.matches(cand.evidence):
                return i
    return None


def _top_figure_ids(candidates: list, k: int) -> list[str]:
    """Distinct figure IDs among the top-k candidates, in rank order."""
    seen: list[str] = []
    for cand in candidates[:k]:
        fid = cand.evidence.figure_id
        if fid is not None and fid not in seen:
            seen.append(fid)
    return seen


def _probe_ollama(host: str) -> bool:
    import httpx

    try:
        r = httpx.get(f"{host}/api/tags", timeout=5.0)
        r.raise_for_status()
        return True
    except Exception:
        return False


def _aggregate(per_query: list[dict], prefix: str, intent: str | None = None) -> dict:
    """Average a metric family over an intent slice.

    Figure Recall@5 is averaged only over queries that actually have figure
    targets, and reported as None when the slice has none — averaging the 1.0
    that the metric returns for "no figure targets" would imply perfect figure
    retrieval where none was attempted. This is the PR2 reporting convention.
    """
    subset = [q for q in per_query if intent is None or q["intent"] == intent]
    if not subset:
        return {
            "sample_count": 0,
            "recall_at_5": 0.0,
            "figure_recall_at_5": None,
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
            "hit_at_5": 0.0,
            "hit_at_1": 0.0,
        }
    n = len(subset)

    def avg(key: str) -> float:
        return round(sum(q[prefix + key] for q in subset) / n, 4)

    fr_key = prefix + "figure_recall_at_5"
    fr_values = [q[fr_key] for q in subset if q.get(fr_key) is not None]
    figure_recall = round(sum(fr_values) / len(fr_values), 4) if fr_values else None

    return {
        "sample_count": n,
        "recall_at_5": avg("recall_at_5"),
        "figure_recall_at_5": figure_recall,
        "mrr": avg("mrr"),
        "ndcg_at_5": avg("ndcg_at_5"),
        "hit_at_5": avg("hit_at_5"),
        "hit_at_1": avg("hit_at_1"),
    }


def _fmt(v: float | None, width: int) -> str:
    return f"{v:{width}.4f}" if v is not None else f"{'N/A':>{width}}"


def _print_row(label: str, m: dict) -> None:
    print(
        f"  {label:16s}  {_fmt(m['recall_at_5'], 8)}  {_fmt(m['figure_recall_at_5'], 13)}"
        f"  {_fmt(m['mrr'], 6)}  {_fmt(m['ndcg_at_5'], 7)}"
        f"  {_fmt(m['hit_at_5'], 6)}  {_fmt(m['hit_at_1'], 6)}  (n={m['sample_count']})"
    )


HEADER = (
    f"  {'':16s}  {'Recall@5':>8}  {'FigRecall@5':>13}  {'MRR':>6}"
    f"  {'nDCG@5':>7}  {'Hit@5':>6}  {'Hit@1':>6}"
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
    from mrta.retrieval.clip_embedder import CLIP_MODEL_ID, CLIPEmbedder
    from mrta.retrieval.image_store import ImageStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="v1")
    args = parser.parse_args()
    cfg = BENCHMARKS[args.benchmark]

    print(f"=== PR3 Evaluation — Direct CLIP Visual Retrieval ({args.benchmark}) ===")
    print()

    if not cfg["clip_index"].exists():
        print(f"ERROR: CLIP index not found at {cfg['clip_index']}")
        print(
            f"Build it first: python scripts/build_clip_image_index.py --benchmark {args.benchmark}"
        )
        sys.exit(1)

    clip = CLIPEmbedder()
    image_store = ImageStore.load(cfg["clip_index"], clip)
    print(f"CLIP model : {CLIP_MODEL_ID}  (dim={clip.dim}, L2-normalized, IndexFlatIP)")
    print(f"CLIP index : {image_store.size} figure record(s)")

    # Experiment B needs the benchmark's caption index, which embeds queries via Ollama.
    caption_store = None
    if cfg["caption_index"].exists() and _probe_ollama(settings.ollama_host):
        from mrta.retrieval.caption_store import CaptionVectorStore
        from mrta.retrieval.embedder import Embedder

        caption_store = CaptionVectorStore.load(
            cfg["caption_index"], Embedder(settings.embedding_model)
        )
        print(f"Caption idx: {caption_store.size} record(s) (for Experiment B)")
    else:
        print("Caption idx: UNAVAILABLE — Experiment B will be skipped")
        print(
            f"             (needs {cfg['caption_index'].name} + Ollama at {settings.ollama_host})"
        )

    # The benchmark's k caps what Recall can show once the pool is small enough.
    if image_store.size <= K:
        print()
        print(f"  NOTE: k={K} >= index size ({image_store.size}). Every search returns")
        print("  every figure, so Recall@5 and Figure Recall@5 are 1.0 by construction")
        print("  and carry no signal. MRR, nDCG@5 and Hit@1 are the rank-sensitive")
        print("  metrics that actually discriminate on this benchmark.")
    else:
        print()
        print(
            f"  Candidate pool: N={image_store.size} vs k={K} "
            f"({image_store.size / K:.1f}x) — Recall@5 is not structurally saturated."
        )

    manifest = _load_json(cfg["manifest"])
    dataset = _load_json(cfg["queries"])
    adapter = EvalAdapter(manifest)
    queries = dataset["queries"]
    intents = sorted(dataset.get("counts_by_intent", {}).keys()) or sorted(
        {q["intent"] for q in queries}
    )

    print()
    print(f"Running {len(queries)} queries at k={K} ...")
    print()

    per_query: list[dict] = []

    for q in queries:
        qid, text, intent = q["query_id"], q["query"], q["intent"]
        targets = _parse_targets(_evidence_list(q))
        has_figs = _has_figure_targets(targets)

        # --- Experiment A: isolated CLIP ---
        clip_hits = image_store.search(text, top_k=K)
        clip_cands = [
            adapter.from_visual_record(rec, score, rank + 1)
            for rank, (rec, score) in enumerate(clip_hits)
        ]

        clip_metrics = {
            "clip_recall_at_5": round(recall_at_k(clip_cands, targets, K), 4),
            "clip_figure_recall_at_5": (
                round(figure_recall_at_k(clip_cands, targets, K), 4) if has_figs else None
            ),
            "clip_mrr": round(mean_reciprocal_rank(clip_cands, targets), 4),
            "clip_ndcg_at_5": round(ndcg_at_k(clip_cands, targets, K), 4),
            "clip_hit_at_5": round(hit_rate_at_k(clip_cands, targets, K), 4),
            "clip_hit_at_1": round(hit_rate_at_k(clip_cands, targets, 1), 4),
        }
        clip_fig_rank = _figure_rank(clip_cands, targets) if has_figs else None

        row: dict = {
            "query_id": qid,
            "query": text,
            "intent": intent,
            "document_id": q.get("document_id"),
            "has_figure_targets": has_figs,
            "expected_canonical_evidence": [
                {
                    "document_id": t.document_id,
                    "page_number": t.page_number,
                    "figure_id": t.figure_id,
                }
                for t in targets
            ],
            **clip_metrics,
            "clip_figure_rank": clip_fig_rank,
            "clip_top_figure_ids": _top_figure_ids(clip_cands, K),
            "clip_retrieved": [
                {
                    "candidate_id": c.candidate_id,
                    "page_number": c.evidence.page_number,
                    "figure_id": c.evidence.figure_id,
                    "score": round(c.score, 4),
                    "rank": c.rank,
                }
                for c in clip_cands
            ],
        }

        # --- Experiment B: caption stream, for comparison ---
        if caption_store is not None:
            cap_hits = caption_store.search_with_scores(text, k=K)
            cap_cands = [
                adapter.from_caption_record(rec, score, rank + 1)
                for rank, (rec, score) in enumerate(cap_hits)
            ]
            cap_fig_rank = _figure_rank(cap_cands, targets) if has_figs else None
            cap_top = _top_figure_ids(cap_cands, K)
            clip_top = row["clip_top_figure_ids"]

            row.update(
                {
                    "caption_recall_at_5": round(recall_at_k(cap_cands, targets, K), 4),
                    "caption_figure_recall_at_5": (
                        round(figure_recall_at_k(cap_cands, targets, K), 4) if has_figs else None
                    ),
                    "caption_mrr": round(mean_reciprocal_rank(cap_cands, targets), 4),
                    "caption_ndcg_at_5": round(ndcg_at_k(cap_cands, targets, K), 4),
                    "caption_hit_at_5": round(hit_rate_at_k(cap_cands, targets, K), 4),
                    "caption_hit_at_1": round(hit_rate_at_k(cap_cands, targets, 1), 4),
                    "caption_figure_rank": cap_fig_rank,
                    "caption_top_figure_ids": cap_top,
                    "figure_id_overlap": sorted(set(cap_top) & set(clip_top)),
                    "figure_id_union": sorted(set(cap_top) | set(clip_top)),
                    "caption_retrieved": [
                        {
                            "candidate_id": c.candidate_id,
                            "page_number": c.evidence.page_number,
                            "figure_id": c.evidence.figure_id,
                            "score": round(c.score, 4),
                            "rank": c.rank,
                        }
                        for c in cap_cands
                    ],
                }
            )

        per_query.append(row)

        mark = "✓" if clip_metrics["clip_hit_at_5"] > 0 else "✗"
        rank_s = str(clip_fig_rank) if clip_fig_rank else "—"
        print(
            f"  {qid} [{intent:14s}] {mark}  MRR={clip_metrics['clip_mrr']:.2f}"
            f"  H@1={clip_metrics['clip_hit_at_1']:.0f}  figRank={rank_s:2s}"
            f"  | {text[:40]!r}"
        )

    # -----------------------------------------------------------------
    # Experiment A aggregate
    # -----------------------------------------------------------------
    print()
    print("=== Experiment A — Isolated CLIP Retrieval ===")
    print(HEADER)
    clip_agg = {"overall": _aggregate(per_query, "clip_")}
    for intent in intents:
        clip_agg[intent] = _aggregate(per_query, "clip_", intent)
    _print_row("Overall", clip_agg["overall"])
    for intent in intents:
        _print_row(intent, clip_agg[intent])

    caption_agg = None
    if caption_store is not None:
        print()
        print(f"=== PR2 Caption Retrieval (re-measured for comparison, {args.benchmark}) ===")
        print(HEADER)
        caption_agg = {"overall": _aggregate(per_query, "caption_")}
        for intent in intents:
            caption_agg[intent] = _aggregate(per_query, "caption_", intent)
        _print_row("Overall", caption_agg["overall"])
        for intent in intents:
            _print_row(intent, caption_agg[intent])

    # -----------------------------------------------------------------
    # Experiment B — complementarity
    # -----------------------------------------------------------------
    complementarity: dict = {"status": "unavailable"}
    if caption_store is not None:
        fig_queries = [r for r in per_query if r["has_figure_targets"]]
        both, cap_only, clip_only, neither = [], [], [], []
        cap_better, clip_better, tied = [], [], []

        for r in fig_queries:
            cap_rank = r.get("caption_figure_rank")
            clip_rank = r.get("clip_figure_rank")
            if cap_rank and clip_rank:
                both.append(r["query_id"])
                if cap_rank < clip_rank:
                    cap_better.append((r["query_id"], cap_rank, clip_rank))
                elif clip_rank < cap_rank:
                    clip_better.append((r["query_id"], cap_rank, clip_rank))
                else:
                    tied.append((r["query_id"], cap_rank))
            elif cap_rank:
                cap_only.append(r["query_id"])
            elif clip_rank:
                clip_only.append(r["query_id"])
            else:
                neither.append(r["query_id"])

        print()
        print("=== Experiment B — Caption vs CLIP Complementarity ===")
        print(f"  Queries with figure targets : {len(fig_queries)}")
        print(f"  Both found the figure       : {len(both)}")
        print(f"  Caption only                : {len(cap_only)}")
        print(f"  CLIP only                   : {len(clip_only)}")
        print(f"  Neither                     : {len(neither)}")
        print()
        print(f"  Caption ranked higher       : {len(cap_better)}")
        for qid, cap_r, clip_r in cap_better[:15]:
            print(f"      {qid}  caption@{cap_r}  clip@{clip_r}")
        if len(cap_better) > 15:
            print(f"      ... and {len(cap_better) - 15} more")
        print(f"  CLIP ranked higher          : {len(clip_better)}")
        for qid, cap_r, clip_r in clip_better[:15]:
            print(f"      {qid}  caption@{cap_r}  clip@{clip_r}")
        if len(clip_better) > 15:
            print(f"      ... and {len(clip_better) - 15} more")
        print(f"  Tied                        : {len(tied)}")

        n_fig = len(fig_queries) or 1
        cap_h1 = sum(r.get("caption_hit_at_1", 0) for r in fig_queries) / n_fig
        clip_h1 = sum(r.get("clip_hit_at_1", 0) for r in fig_queries) / n_fig
        overlap_sizes = [len(r.get("figure_id_overlap", [])) for r in fig_queries]
        union_sizes = [len(r.get("figure_id_union", [])) for r in fig_queries]
        mean_jaccard = (
            sum(o / u for o, u in zip(overlap_sizes, union_sizes, strict=True) if u) / n_fig
        )

        print()
        print(f"  Caption Hit@1 (figure queries): {cap_h1:.4f}")
        print(f"  CLIP    Hit@1 (figure queries): {clip_h1:.4f}")
        print(f"  Mean top-{K} figure-ID Jaccard : {mean_jaccard:.4f}")

        complementarity = {
            "status": "measured",
            "n_figure_queries": len(fig_queries),
            "both_found": both,
            "caption_only": cap_only,
            "clip_only": clip_only,
            "neither": neither,
            "caption_ranked_higher": [
                {"query_id": q, "caption_rank": cr, "clip_rank": lr} for q, cr, lr in cap_better
            ],
            "clip_ranked_higher": [
                {"query_id": q, "caption_rank": cr, "clip_rank": lr} for q, cr, lr in clip_better
            ],
            "tied": [{"query_id": q, "rank": cr} for q, cr in tied],
            "caption_hit_at_1": round(cap_h1, 4),
            "clip_hit_at_1": round(clip_h1, 4),
            "mean_figure_id_jaccard": round(mean_jaccard, 4),
        }

    # -----------------------------------------------------------------
    # Comparison table (PR1 loaded from its own result file, never hardcoded)
    # -----------------------------------------------------------------
    pr1_metrics: dict | None = None
    if cfg["pr1_results"].exists():
        pr1_metrics = _load_json(cfg["pr1_results"])
    else:
        print(
            f"\n  NOTE: PR1 results not found at {cfg['pr1_results']}; skipping comparison table."
        )

    if pr1_metrics is not None:
        pr1m = pr1_metrics["metrics"]
        print()
        print(f"=== PR1 → PR2 → PR3 Comparison ({args.benchmark}) ===")
        print(f"  {'':26s}  {'PR1 text':>9}  {'PR2 caption':>12}  {'PR3 CLIP':>9}")

        def cap_val(slice_name: str, key: str):
            return caption_agg[slice_name][key] if caption_agg else None

        rows = [
            (
                "Figure Recall@5",
                None,
                cap_val("overall", "figure_recall_at_5"),
                clip_agg["overall"]["figure_recall_at_5"],
            ),
        ]
        for intent in intents:
            rows.append(
                (
                    f"{intent} Recall@5",
                    pr1m.get(intent, {}).get("recall_at_5"),
                    cap_val(intent, "recall_at_5"),
                    clip_agg[intent]["recall_at_5"],
                )
            )
        for intent in intents:
            rows.append(
                (
                    f"{intent} MRR",
                    pr1m.get(intent, {}).get("mrr"),
                    cap_val(intent, "mrr"),
                    clip_agg[intent]["mrr"],
                )
            )
        for label, a, b, c in rows:
            print(f"  {label:26s}  {_fmt(a, 9)}  {_fmt(b, 12)}  {_fmt(c, 9)}")

    # -----------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "status": "measured",
        "benchmark": args.benchmark,
        "model": CLIP_MODEL_ID,
        "embedding_dimension": clip.dim,
        "normalization": "L2",
        "similarity": "inner_product/cosine",
        "k": K,
        "dataset_version": dataset["dataset_version"],
        "corpus_version": dataset["corpus_version"],
        "index_size": image_store.size,
        "candidate_pool_note": (
            f"k={K}, N_visual={image_store.size} (ratio {image_store.size / K:.2f}x). "
            + (
                "N <= k: Recall@5/Figure Recall@5 are 1.0 by construction, no signal."
                if image_store.size <= K
                else "N > k: Recall@5 is not structurally saturated on this benchmark."
            )
        ),
        "score_pooling": (
            "NOT PERFORMED. CLIP (512-D CLIP space) and nomic-embed-text scores are "
            "uncalibrated with respect to each other; pooling raw cosines would be "
            "meaningless. Cross-stream combination is deferred to PR4 (RRF)."
        ),
        "pr1_baseline_source": str(cfg["pr1_results"].relative_to(REPO_ROOT)),
        "clip_only": clip_agg,
        "caption_only": caption_agg,
        "complementarity": complementarity,
    }

    metrics_path = cfg["results_dir"] / cfg["metrics_filename"]
    per_query_path = cfg["results_dir"] / cfg["per_query_filename"]
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query, indent=2), encoding="utf-8")

    print()
    print(f"Saved → {metrics_path}")
    print(f"Saved → {per_query_path}")


if __name__ == "__main__":
    main()
