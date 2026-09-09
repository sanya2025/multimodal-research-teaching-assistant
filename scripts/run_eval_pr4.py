"""PR4 evaluation driver — equal-weight RRF fusion over text, caption and CLIP.

Runs five configurations over the frozen v2 benchmark:

    A  text                      (control: RRF over one stream must not perturb it)
    B  text + caption
    C  text + CLIP
    D  caption + CLIP
    E  text + caption + CLIP     (main system)

The headline ablation is B vs E: does CLIP add marginal value once caption
retrieval is already present?

Three different k's are kept deliberately distinct (spec §7):

    CANDIDATE_POOL_SIZE = 20   how deep each stream retrieves before fusion
    RRF_K               = 60   the RRF smoothing constant
    EVALUATION_K        = 5    the metric cutoff

Streams are NOT truncated to EVALUATION_K before fusion — fusion needs depth to
be able to promote evidence that sat below rank 5 in every individual stream.

Raw similarity scores are never pooled. Text/caption use nomic-embed-text and
CLIP uses a different 512-D space; their cosines are not calibrated against each
other. Fusion is purely rank-based.

Usage:
    python scripts/run_eval_pr4.py

Requirements:
    - Ollama running with nomic-embed-text (text + caption query embedding)
    - data/vector_store/v2_corpus/            (build_text_index.py --benchmark v2)
    - data/eval/indices/v2/caption_index/     (build_caption_index.py --benchmark v2)
    - data/eval/indices/v2/clip_image_index/  (build_clip_image_index.py --benchmark v2)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Three distinct k concepts — never conflate (spec §7).
CANDIDATE_POOL_SIZE = 20
RRF_K = 60
EVALUATION_K = 5

VECTOR_STORE_PATH = REPO_ROOT / "data" / "vector_store" / "v2_corpus"
CAPTION_INDEX_PATH = REPO_ROOT / "data" / "eval" / "indices" / "v2" / "caption_index"
CLIP_INDEX_PATH = REPO_ROOT / "data" / "eval" / "indices" / "v2" / "clip_image_index"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json"
QUERIES_PATH = REPO_ROOT / "data" / "eval" / "queries_v2.json"
RESULTS_DIR = REPO_ROOT / "results" / "v2"

PR3_PER_QUERY_PATH = RESULTS_DIR / "pr3_per_query.json"

# The five required ablations, as stream-name tuples.
CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "rrf_text": ("text",),
    "rrf_text_caption": ("text", "caption"),
    "rrf_text_clip": ("text", "clip"),
    "rrf_caption_clip": ("caption", "clip"),
    "rrf_text_caption_clip": ("text", "caption", "clip"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_list(query: dict) -> list[dict]:
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
    return any(t.figure_id is not None for t in targets)


def _probe_ollama(host: str) -> None:
    import httpx

    try:
        r = httpx.get(f"{host}/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {host}: {e}")
        sys.exit(1)


def _target_rank(candidates: list, targets: list, figures_only: bool = False) -> int | None:
    """1-based rank of the first candidate matching any target, else None."""
    wanted = [t for t in targets if t.figure_id is not None] if figures_only else list(targets)
    for i, cand in enumerate(candidates, start=1):
        for target in wanted:
            if target.matches(cand.evidence):
                return i
    return None


def _hybrid_both_covered(candidates: list, targets: list, k: int) -> bool | None:
    """True if BOTH a text and a figure target appear within the top-k.

    Returns None when the query does not have both kinds of target, so the
    metric is reported as N/A rather than silently counted as a success.
    """
    figure_targets = [t for t in targets if t.figure_id is not None]
    text_targets = [t for t in targets if t.figure_id is None]
    if not figure_targets or not text_targets:
        return None
    top = candidates[:k]
    fig_hit = any(t.matches(c.evidence) for c in top for t in figure_targets)
    txt_hit = any(t.matches(c.evidence) for c in top for t in text_targets)
    return fig_hit and txt_hit


def _metrics_for(candidates: list, targets: list, has_figs: bool) -> dict:
    from mrta.eval.retrieval_metrics import (
        figure_recall_at_k,
        hit_rate_at_k,
        mean_reciprocal_rank,
        ndcg_at_k,
        recall_at_k,
    )

    return {
        "recall_at_1": round(recall_at_k(candidates, targets, 1), 4),
        "recall_at_5": round(recall_at_k(candidates, targets, EVALUATION_K), 4),
        "hit_at_1": round(hit_rate_at_k(candidates, targets, 1), 4),
        "hit_at_5": round(hit_rate_at_k(candidates, targets, EVALUATION_K), 4),
        "mrr": round(mean_reciprocal_rank(candidates, targets), 4),
        "ndcg_at_5": round(ndcg_at_k(candidates, targets, EVALUATION_K), 4),
        "figure_recall_at_1": (
            round(figure_recall_at_k(candidates, targets, 1), 4) if has_figs else None
        ),
        "figure_recall_at_5": (
            round(figure_recall_at_k(candidates, targets, EVALUATION_K), 4) if has_figs else None
        ),
    }


_METRIC_KEYS = (
    "recall_at_1",
    "recall_at_5",
    "hit_at_1",
    "hit_at_5",
    "mrr",
    "ndcg_at_5",
    "figure_recall_at_1",
    "figure_recall_at_5",
)


def _aggregate(rows: list[dict], config: str) -> dict:
    """Average one configuration's metrics over a set of per-query rows.

    Figure Recall is averaged only over queries that actually have figure
    targets and reported as None for slices with none — the PR2/PR3 convention.
    """
    if not rows:
        return {"sample_count": 0, **{k: None for k in _METRIC_KEYS}}
    out: dict = {"sample_count": len(rows)}
    for key in _METRIC_KEYS:
        values = [r["configs"][config][key] for r in rows]
        present = [v for v in values if v is not None]
        out[key] = round(sum(present) / len(present), 4) if present else None
    # hybrid both-target coverage
    both = [r["hybrid_both_covered"][config] for r in rows]
    both_present = [v for v in both if v is not None]
    out["both_target_recall_at_5"] = (
        round(sum(1 for v in both_present if v) / len(both_present), 4) if both_present else None
    )
    return out


def _fmt(v, width: int = 8) -> str:
    return f"{v:{width}.4f}" if isinstance(v, (int, float)) else f"{'N/A':>{width}}"


def _print_metric_table(title: str, agg_by_config: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(
        f"  {'configuration':24s} {'R@1':>7} {'R@5':>7} {'H@1':>7} {'H@5':>7} "
        f"{'MRR':>7} {'nDCG@5':>7} {'FigR@1':>7} {'FigR@5':>7}"
    )
    for config in CONFIGURATIONS:
        m = agg_by_config[config]
        print(
            f"  {config:24s} {_fmt(m['recall_at_1'],7)} {_fmt(m['recall_at_5'],7)} "
            f"{_fmt(m['hit_at_1'],7)} {_fmt(m['hit_at_5'],7)} {_fmt(m['mrr'],7)} "
            f"{_fmt(m['ndcg_at_5'],7)} {_fmt(m['figure_recall_at_1'],7)} "
            f"{_fmt(m['figure_recall_at_5'],7)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from mrta.core.config import settings
    from mrta.eval.adapter import EvalAdapter
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.clip_embedder import CLIPEmbedder
    from mrta.retrieval.embedder import Embedder
    from mrta.retrieval.fusion import canonical_identity, reciprocal_rank_fusion_canonical
    from mrta.retrieval.image_store import ImageStore
    from mrta.retrieval.vector_store import VectorStore

    print("=== PR4 Evaluation — Equal-Weight RRF Fusion (v2) ===\n")
    print(f"candidate_pool_size = {CANDIDATE_POOL_SIZE}")
    print(f"rrf_k               = {RRF_K}")
    print(f"evaluation_k        = {EVALUATION_K}")
    print("weights             = text 1.0 / caption 1.0 / clip 1.0 (fixed)\n")

    for path, what in (
        (VECTOR_STORE_PATH, "text vector store"),
        (CAPTION_INDEX_PATH, "caption index"),
        (CLIP_INDEX_PATH, "CLIP index"),
    ):
        if not path.exists():
            print(f"ERROR: {what} not found at {path}")
            sys.exit(1)

    _probe_ollama(settings.ollama_host)

    # CLIP index first: ImageStore.load() warms torch before FAISS, avoiding the
    # macOS libomp conflict documented in CLIPEmbedder.warmup().
    clip = CLIPEmbedder()
    image_store = ImageStore.load(CLIP_INDEX_PATH, clip)
    embedder = Embedder(settings.embedding_model)
    text_store = VectorStore.load(VECTOR_STORE_PATH, embedder)
    caption_store = CaptionVectorStore.load(CAPTION_INDEX_PATH, embedder)

    print(f"text index    : {len(text_store._chunks)} chunks")
    print(f"caption index : {caption_store.size} records")
    print(f"CLIP index    : {image_store.size} records")

    manifest = _load_json(MANIFEST_PATH)
    dataset = _load_json(QUERIES_PATH)
    adapter = EvalAdapter(manifest)
    queries = dataset["queries"]

    # PR3's per-query complementarity groups, for the recovery analysis (spec §19).
    pr3_groups: dict[str, str] = {}
    if PR3_PER_QUERY_PATH.exists():
        for row in _load_json(PR3_PER_QUERY_PATH):
            if not row.get("has_figure_targets"):
                continue
            cap_rank = row.get("caption_figure_rank")
            clip_rank = row.get("clip_figure_rank")
            if cap_rank and clip_rank:
                pr3_groups[row["query_id"]] = "both"
            elif cap_rank:
                pr3_groups[row["query_id"]] = "caption_only"
            elif clip_rank:
                pr3_groups[row["query_id"]] = "clip_only"
            else:
                pr3_groups[row["query_id"]] = "neither"

    print(f"\nRunning {len(queries)} queries " f"({len(CONFIGURATIONS)} configurations each) ...\n")

    per_query: list[dict] = []

    for q in queries:
        qid, text, intent = q["query_id"], q["query"], q["intent"]
        targets = _parse_targets(_evidence_list(q))
        has_figs = _has_figure_targets(targets)

        # --- retrieve candidate pools (depth 20, NOT the eval cutoff) ---
        text_cands = [
            adapter.chunk_to_candidate(chunk, score, i + 1)
            for i, (chunk, score) in enumerate(
                text_store.search_with_scores(text, k=CANDIDATE_POOL_SIZE)
            )
        ]
        caption_cands = [
            adapter.from_caption_record(rec, score, i + 1)
            for i, (rec, score) in enumerate(
                caption_store.search_with_scores(text, k=CANDIDATE_POOL_SIZE)
            )
        ]
        clip_cands = [
            adapter.from_visual_record(rec, score, i + 1)
            for i, (rec, score) in enumerate(image_store.search(text, top_k=CANDIDATE_POOL_SIZE))
        ]

        stream_pool = {"text": text_cands, "caption": caption_cands, "clip": clip_cands}

        # Payload provenance for merged figures (diagnostics only).
        payloads: dict[str, dict[str, dict]] = {"caption": {}, "clip": {}}
        for c in caption_cands:
            payloads["caption"]["|".join(canonical_identity(c))] = {"stream": "caption"}
        for c in clip_cands:
            payloads["clip"]["|".join(canonical_identity(c))] = {"stream": "clip"}

        # --- per-stream target ranks within the pool (for §19 analysis) ---
        pool_ranks = {
            name: _target_rank(cands, targets, figures_only=has_figs)
            for name, cands in stream_pool.items()
        }

        row: dict = {
            "query_id": qid,
            "query": text,
            "intent": intent,
            "paper": q.get("document_id"),
            "retrieval_challenge": q.get("retrieval_challenge"),
            "difficulty": q.get("difficulty"),
            "has_figure_targets": has_figs,
            "pr3_group": pr3_groups.get(qid),
            "expected_canonical_evidence": [
                {
                    "document_id": t.document_id,
                    "page_number": t.page_number,
                    "figure_id": t.figure_id,
                }
                for t in targets
            ],
            "stream_pool_target_rank": pool_ranks,
            "stream_candidates": {
                name: [
                    {
                        "canonical_id": "|".join(canonical_identity(c)),
                        "rank": c.rank,
                        "raw_score": round(c.score, 4),  # diagnostic only
                        "figure_id": c.evidence.figure_id,
                        "page": c.evidence.page_number,
                        "document_id": c.evidence.document_id,
                    }
                    for c in cands[:EVALUATION_K]
                ]
                for name, cands in stream_pool.items()
            },
            "configs": {},
            "fused_target_rank": {},
            "hybrid_both_covered": {},
            "fused_top5": {},
        }

        # --- run each configuration ---
        for config, stream_names in CONFIGURATIONS.items():
            streams = {name: stream_pool[name] for name in stream_names}
            fused = reciprocal_rank_fusion_canonical(
                streams, k=RRF_K, top_k=None, payloads=payloads
            )
            fused_candidates = [
                adapter.from_fused_candidate(fc, rank=i + 1) for i, fc in enumerate(fused)
            ]

            # Metrics are computed on the top-EVALUATION_K slice, because that is
            # exactly the protocol PR1-PR3 used (they retrieved k=5 and scored the
            # resulting 5-item list). mean_reciprocal_rank has no k parameter and
            # scans whatever list it is given, so passing the full depth-20 fused
            # list here would silently compute MRR@20 and make PR4 non-comparable
            # with the frozen baselines. The full list is kept for rank diagnostics.
            fused_eval = fused_candidates[:EVALUATION_K]

            row["configs"][config] = _metrics_for(fused_eval, targets, has_figs)
            row["fused_target_rank"][config] = _target_rank(
                fused_candidates, targets, figures_only=has_figs
            )
            row["hybrid_both_covered"][config] = _hybrid_both_covered(
                fused_eval, targets, EVALUATION_K
            )
            row["fused_top5"][config] = [
                {
                    "canonical_id": fc.canonical_id,
                    "rrf_score": round(fc.score, 6),
                    "modality_sources": list(fc.modality_sources),
                    "source_ranks": fc.source_ranks,
                    "figure_id": fc.figure_id,
                }
                for fc in fused[:EVALUATION_K]
            ]

        per_query.append(row)

        main_cfg = row["configs"]["rrf_text_caption_clip"]
        mark = "✓" if main_cfg["hit_at_5"] > 0 else "✗"
        print(
            f"  {qid} [{intent:14s}] {mark} "
            f"R@5={main_cfg['recall_at_5']:.2f} MRR={main_cfg['mrr']:.2f} "
            f"| {text[:38]!r}"
        )

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------
    intents = sorted({r["intent"] for r in per_query})
    papers = sorted({r["paper"] for r in per_query if r["paper"]})
    challenges = sorted({r["retrieval_challenge"] for r in per_query if r["retrieval_challenge"]})

    overall = {c: _aggregate(per_query, c) for c in CONFIGURATIONS}
    by_intent = {
        i: {c: _aggregate([r for r in per_query if r["intent"] == i], c) for c in CONFIGURATIONS}
        for i in intents
    }
    by_paper = {
        p: {c: _aggregate([r for r in per_query if r["paper"] == p], c) for c in CONFIGURATIONS}
        for p in papers
    }
    by_challenge = {
        ch: {
            c: _aggregate([r for r in per_query if r["retrieval_challenge"] == ch], c)
            for c in CONFIGURATIONS
        }
        for ch in challenges
    }

    _print_metric_table("=== Overall (n=100) ===", overall)
    for i in intents:
        n = by_intent[i]["rrf_text"]["sample_count"]
        _print_metric_table(f"=== {i} (n={n}) ===", by_intent[i])

    # -----------------------------------------------------------------
    # CLIP marginal-value delta table (spec §31)
    # -----------------------------------------------------------------
    print("\n\n=== CLIP marginal value: Text+Caption  vs  Text+Caption+CLIP ===")
    print(f"  {'metric / slice':34s} {'T+C':>9} {'T+C+CLIP':>9} {'Δ CLIP':>9}")

    def _delta_row(label: str, slice_agg: dict, key: str) -> None:
        b = slice_agg["rrf_text_caption"][key]
        e = slice_agg["rrf_text_caption_clip"][key]
        if b is None or e is None:
            print(f"  {label:34s} {_fmt(b,9)} {_fmt(e,9)} {'N/A':>9}")
        else:
            print(f"  {label:34s} {b:9.4f} {e:9.4f} {e - b:+9.4f}")

    for key, lbl in (
        ("mrr", "Overall MRR"),
        ("recall_at_5", "Overall Recall@5"),
        ("figure_recall_at_5", "Overall Figure Recall@5"),
    ):
        _delta_row(lbl, overall, key)
    for i in intents:
        _delta_row(f"{i} MRR", by_intent[i], "mrr")
    for i in intents:
        _delta_row(f"{i} FigR@5", by_intent[i], "figure_recall_at_5")

    # -----------------------------------------------------------------
    # Per-paper
    # -----------------------------------------------------------------
    print("\n\n=== Per-paper (Recall@5 / MRR) ===")
    print(f"  {'paper':28s} " + " ".join(f"{c.replace('rrf_',''):>22s}" for c in CONFIGURATIONS))
    for p in papers:
        cells = " ".join(
            f"{by_paper[p][c]['recall_at_5']:10.4f}/{by_paper[p][c]['mrr']:<11.4f}"
            for c in CONFIGURATIONS
        )
        print(f"  {p:28s} {cells}")

    # -----------------------------------------------------------------
    # Candidate recovery / agreement analysis (spec §19, §32)
    # -----------------------------------------------------------------
    fig_rows = [r for r in per_query if r["has_figure_targets"]]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in fig_rows:
        groups[r["pr3_group"] or "unknown"].append(r)

    print("\n\n=== Complementarity groups (from PR3) under fusion ===")
    print(
        f"  {'group':14s} {'n':>4} {'capOnlyFigR@5':>14} "
        f"{'T+C FigR@5':>11} {'T+C+CLIP FigR@5':>16}"
    )
    group_summary: dict[str, dict] = {}
    for name in ("both", "caption_only", "clip_only", "neither", "unknown"):
        rows = groups.get(name, [])
        if not rows:
            continue
        tc = _aggregate(rows, "rrf_text_caption")
        tcc = _aggregate(rows, "rrf_text_caption_clip")
        cc = _aggregate(rows, "rrf_caption_clip")
        print(
            f"  {name:14s} {len(rows):4d} {_fmt(cc['figure_recall_at_5'],14)} "
            f"{_fmt(tc['figure_recall_at_5'],11)} {_fmt(tcc['figure_recall_at_5'],16)}"
        )
        group_summary[name] = {
            "n": len(rows),
            "query_ids": [r["query_id"] for r in rows],
            "caption_clip_figure_recall_at_5": cc["figure_recall_at_5"],
            "text_caption_figure_recall_at_5": tc["figure_recall_at_5"],
            "text_caption_clip_figure_recall_at_5": tcc["figure_recall_at_5"],
        }

    # Candidate-generation failure vs ranking/fusion opportunity, for "neither".
    print("\n=== 'neither' group: candidate-generation vs fusion opportunity ===")
    gen_failures, fusion_opportunities = [], []
    for r in groups.get("neither", []):
        in_pool = any(v is not None for v in r["stream_pool_target_rank"].values())
        (fusion_opportunities if in_pool else gen_failures).append(r["query_id"])
    print(f"  candidate-generation failure (target absent from every pool): {len(gen_failures)}")
    print(f"    {gen_failures}")
    print(
        f"  ranking/fusion opportunity  (target in some pool, below top-5): "
        f"{len(fusion_opportunities)}"
    )
    print(f"    {fusion_opportunities}")

    recovered = [
        r["query_id"]
        for r in groups.get("neither", [])
        if (rank := r["fused_target_rank"]["rrf_text_caption_clip"]) is not None
        and rank <= EVALUATION_K
    ]
    print(f"  of which fusion actually recovered into top-5: {len(recovered)} {recovered}")

    # Target-rank movement: best single stream -> fused (three-stream).
    print("\n=== Target-rank movement (figure queries, best single stream -> fused) ===")
    before_ranks, after_ranks = [], []
    promoted_top5 = promoted_top1 = pushed_out = 0
    for r in fig_rows:
        pool = [v for v in r["stream_pool_target_rank"].values() if v is not None]
        before = min(pool) if pool else None
        after = r["fused_target_rank"]["rrf_text_caption_clip"]
        if before is not None and after is not None:
            before_ranks.append(before)
            after_ranks.append(after)
            if before > EVALUATION_K >= after:
                promoted_top5 += 1
            if before > 1 == after:
                promoted_top1 += 1
            if before <= EVALUATION_K < after:
                pushed_out += 1
    if before_ranks:
        print(
            f"  mean target rank before fusion (best stream): "
            f"{sum(before_ranks)/len(before_ranks):.2f}"
        )
        print(
            f"  mean target rank after fusion  (3-stream)   : "
            f"{sum(after_ranks)/len(after_ranks):.2f}"
        )
    print(f"  promoted into top-5 : {promoted_top5}")
    print(f"  promoted into top-1 : {promoted_top1}")
    print(f"  pushed out of top-5 : {pushed_out}")

    # -----------------------------------------------------------------
    # Pool-saturation diagnostic (NOT a tuning sweep — see note below)
    # -----------------------------------------------------------------
    # The primary PR4 result above is fixed at CANDIDATE_POOL_SIZE=20 and is the
    # only configuration reported as a result. This diagnostic exists solely to
    # explain *why* three-stream fusion behaves as it does: the corpus contains
    # only 21 canonical figures, so a depth-20 visual pool returns most of them
    # for any query, making "caption and CLIP agree" — the signal RRF rewards —
    # near-universal and therefore uninformative. No configuration is selected
    # on the basis of these numbers.
    total_canonical_figures = len(
        {(d["document_id"], f["figure_id"]) for d in manifest["documents"] for f in d["figures"]}
    )
    print("\n\n=== Pool-saturation diagnostic (explanatory, not a tuning sweep) ===")
    print(f"  canonical figures in corpus: {total_canonical_figures}")
    print(
        f"  {'depth':>6} {'caption figs':>13} {'clip figs':>10} {'in BOTH':>8} {'% of corpus':>12}"
    )
    saturation: list[dict] = []
    probe_queries = [q["query"] for q in queries[:20]]
    for depth in (5, 10, 20):
        both_counts = []
        for pq in probe_queries:
            cap_ids = {
                "|".join(canonical_identity(adapter.from_caption_record(r, s, i + 1)))
                for i, (r, s) in enumerate(caption_store.search_with_scores(pq, k=depth))
            }
            clip_ids = {
                "|".join(canonical_identity(adapter.from_visual_record(r, s, i + 1)))
                for i, (r, s) in enumerate(image_store.search(pq, top_k=depth))
            }
            both_counts.append((len(cap_ids), len(clip_ids), len(cap_ids & clip_ids)))
        mean_cap = sum(c[0] for c in both_counts) / len(both_counts)
        mean_clip = sum(c[1] for c in both_counts) / len(both_counts)
        mean_both = sum(c[2] for c in both_counts) / len(both_counts)
        pct = mean_both / total_canonical_figures
        print(f"  {depth:>6} {mean_cap:>13.1f} {mean_clip:>10.1f} {mean_both:>8.1f} {pct:>11.0%}")
        saturation.append(
            {
                "depth": depth,
                "mean_caption_distinct_figures": round(mean_cap, 2),
                "mean_clip_distinct_figures": round(mean_clip, 2),
                "mean_figures_in_both_streams": round(mean_both, 2),
                "fraction_of_corpus_figures_agreed": round(pct, 4),
            }
        )
    print("  (averaged over the first 20 benchmark queries)")

    # -----------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "status": "measured",
        "benchmark": "v2",
        "dataset_version": dataset["dataset_version"],
        "corpus_version": dataset["corpus_version"],
        "rrf_k": RRF_K,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "evaluation_k": EVALUATION_K,
        "weights": {"text": 1.0, "caption": 1.0, "clip": 1.0},
        "weights_note": "Equal and fixed. No weight or k sweep was run against v2.",
        "streams": {
            "text": f"VectorStore v2_corpus ({len(text_store._chunks)} chunks, nomic-embed-text)",
            "caption": f"CaptionVectorStore v2 ({caption_store.size} records, nomic-embed-text)",
            "clip": f"ImageStore v2 ({image_store.size} records, openai/clip-vit-base-patch32)",
        },
        "score_pooling": (
            "NOT PERFORMED. Fusion is rank-based only; raw cosines from "
            "nomic-embed-text and CLIP are never compared or pooled."
        ),
        "configurations": {c: list(s) for c, s in CONFIGURATIONS.items()},
        "overall": overall,
        "by_intent": by_intent,
        "by_paper": by_paper,
        "by_retrieval_challenge": by_challenge,
        "complementarity_groups": group_summary,
        "neither_group_breakdown": {
            "candidate_generation_failure": gen_failures,
            "ranking_fusion_opportunity": fusion_opportunities,
            "recovered_into_top5_by_fusion": recovered,
        },
        "metric_depth_note": (
            f"All metrics are computed on the top-{EVALUATION_K} fused slice, matching "
            "the protocol PR1-PR3 used (they retrieved k=5 and scored that 5-item list). "
            "mean_reciprocal_rank takes no k argument, so scoring the full depth-"
            f"{CANDIDATE_POOL_SIZE} fused list would yield MRR@{CANDIDATE_POOL_SIZE} and "
            "break comparability with the frozen baselines."
        ),
        "pool_saturation_diagnostic": {
            "note": (
                "Explanatory only — no configuration was selected from these numbers. "
                "The primary result is fixed at candidate_pool_size=20 per the PR4 spec."
            ),
            "canonical_figures_in_corpus": total_canonical_figures,
            "by_depth": saturation,
        },
        "rank_movement": {
            "mean_target_rank_before_fusion": (
                round(sum(before_ranks) / len(before_ranks), 4) if before_ranks else None
            ),
            "mean_target_rank_after_fusion": (
                round(sum(after_ranks) / len(after_ranks), 4) if after_ranks else None
            ),
            "promoted_into_top5": promoted_top5,
            "promoted_into_top1": promoted_top1,
            "pushed_out_of_top5": pushed_out,
        },
    }

    metrics_path = RESULTS_DIR / "pr4_rrf_metrics.json"
    per_query_path = RESULTS_DIR / "pr4_per_query.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query, indent=2), encoding="utf-8")

    print(f"\nSaved → {metrics_path}")
    print(f"Saved → {per_query_path}")


if __name__ == "__main__":
    main()
