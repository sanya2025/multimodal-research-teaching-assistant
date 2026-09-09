"""PR5 evaluation driver — cross-encoder reranking over RRF-fused candidates.

Pipeline under test::

    Text ──────┐
    Caption ───┼→ canonical RRF → top-20 → CrossEncoder → top-5
    CLIP ──────┘

Four configurations are measured on the frozen v2 benchmark:

    rrf_text_caption            RRF T+C                 (PR4 baseline, recomputed)
    rrf_text_caption_clip       RRF T+C+CLIP            (PR4 baseline, recomputed)
    rerank_text_caption         RRF T+C      → CE       (PR5)
    rerank_text_caption_clip    RRF T+C+CLIP → CE       (PR5, main system)

The headline question is whether a query-aware reranker lets CLIP's extra visual
recall be used without the ranking collapse three-stream RRF caused in PR4.

Four distinct k's are kept deliberately separate:

    CANDIDATE_POOL_SIZE = 20   how deep each stream retrieves before fusion
    RRF_K               = 60   the RRF smoothing constant
    RERANK_TOP_N        = 20   fused candidates entering the cross-encoder
    EVALUATION_K        = 5    the metric cutoff

None of them is tuned against v2. They are fixed at the PR4 values so PR5 stays
directly comparable with the frozen baselines.

The cross-encoder is a TEXT model. Figure candidates are reranked through their
production-derived textual representation only — see mrta.retrieval.reranker.

Usage:
    python scripts/run_eval_pr5.py

Requirements:
    - Ollama running with nomic-embed-text (text + caption query embedding)
    - data/vector_store/v2_corpus/            (build_text_index.py --benchmark v2)
    - data/eval/indices/v2/caption_index/     (build_caption_index.py --benchmark v2)
    - data/eval/indices/v2/clip_image_index/  (build_clip_image_index.py --benchmark v2)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (downloads from HuggingFace on first run)
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Four distinct k concepts — never conflate.
CANDIDATE_POOL_SIZE = 20
RRF_K = 60
RERANK_TOP_N = 20
EVALUATION_K = 5

VECTOR_STORE_PATH = REPO_ROOT / "data" / "vector_store" / "v2_corpus"
CAPTION_INDEX_PATH = REPO_ROOT / "data" / "eval" / "indices" / "v2" / "caption_index"
CLIP_INDEX_PATH = REPO_ROOT / "data" / "eval" / "indices" / "v2" / "clip_image_index"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json"
QUERIES_PATH = REPO_ROOT / "data" / "eval" / "queries_v2.json"
RESULTS_DIR = REPO_ROOT / "results" / "v2"

PR1_METRICS_PATH = RESULTS_DIR / "pr1_baseline_metrics.json"
PR1_PER_QUERY_PATH = RESULTS_DIR / "pr1_per_query.json"
PR2_METRICS_PATH = RESULTS_DIR / "pr2_caption_metrics.json"
PR2_PER_QUERY_PATH = RESULTS_DIR / "pr2_per_query.json"
PR3_METRICS_PATH = RESULTS_DIR / "pr3_clip_metrics.json"
PR4_METRICS_PATH = RESULTS_DIR / "pr4_rrf_metrics.json"
PR4_PER_QUERY_PATH = RESULTS_DIR / "pr4_per_query.json"

# RRF baselines recomputed here, then cross-checked against the frozen PR4 artifact.
RRF_CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "rrf_text_caption": ("text", "caption"),
    "rrf_text_caption_clip": ("text", "caption", "clip"),
}
# Each RRF config gets a reranked counterpart over the same fused pool.
RERANKED_OF = {
    "rrf_text_caption": "rerank_text_caption",
    "rrf_text_caption_clip": "rerank_text_caption_clip",
}
ALL_CONFIGS: tuple[str, ...] = (
    "rrf_text_caption",
    "rrf_text_caption_clip",
    "rerank_text_caption",
    "rerank_text_caption_clip",
)


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


def _target_rank(candidates: list, targets: list, kind: str = "any") -> int | None:
    """1-based rank of the first candidate matching a target of the given kind.

    kind: "any" | "figure" (figure_id set) | "text" (figure_id None).
    Returns None when no such target exists or none is found.
    """
    if kind == "figure":
        wanted = [t for t in targets if t.figure_id is not None]
    elif kind == "text":
        wanted = [t for t in targets if t.figure_id is None]
    else:
        wanted = list(targets)
    if not wanted:
        return None
    for i, cand in enumerate(candidates, start=1):
        for target in wanted:
            if target.matches(cand.evidence):
                return i
    return None


def _hybrid_both_covered(candidates: list, targets: list, k: int) -> bool | None:
    """True if BOTH a text and a figure target appear within the top-k.

    None when the query lacks one of the two kinds, so it is reported as N/A
    rather than silently counted as a success. Text and figure evidence are
    never collapsed into page-level relevance.
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
    """The PR1-PR4 metric set, computed on whatever slice is passed in.

    Callers MUST pass the top-EVALUATION_K slice: mean_reciprocal_rank takes no
    k argument and scans the whole list, so handing it the depth-20 pool would
    silently produce MRR@20 and break comparability with the frozen baselines.
    """
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
    targets and reported as None for slices with none — the PR2-PR4 convention.
    """
    if not rows:
        return {"sample_count": 0, **{k: None for k in _METRIC_KEYS}}
    out: dict = {"sample_count": len(rows)}
    for key in _METRIC_KEYS:
        values = [r["configs"][config][key] for r in rows]
        present = [v for v in values if v is not None]
        out[key] = round(sum(present) / len(present), 4) if present else None
    both = [r["hybrid_both_covered"][config] for r in rows]
    both_present = [v for v in both if v is not None]
    out["both_target_recall_at_5"] = (
        round(sum(1 for v in both_present if v) / len(both_present), 4) if both_present else None
    )
    return out


def _fmt(v, width: int = 8) -> str:
    return f"{v:{width}.4f}" if isinstance(v, (int, float)) else f"{'N/A':>{width}}"


# ---------------------------------------------------------------------------
# Figure text lookup — built once from the frozen PR2 caption index
# ---------------------------------------------------------------------------


def _build_figure_text_lookup(caption_records: list, adapter) -> tuple[dict, dict]:
    """Map canonical_id -> reranker payload for every figure in the corpus.

    Built from the frozen PR2 caption index, which is production-derived: VLM
    captions and descriptions written by VisualAnalyzer at index-build time, plus
    the nearby-text fallback captured at figure-extraction time. No benchmark
    annotation is read, so no ground truth can reach the model.

    One canonical figure may own several caption records (multi-crop figures).
    Selection is deterministic: prefer a record carrying a VLM caption, then the
    lowest evidence_id. This makes the representation identical whether the
    figure entered via the caption stream, the CLIP stream, or both — CLIP's
    VisualRecord carries no text of its own, so without this lookup a CLIP-only
    figure would have no textual representation at all.

    Returns (payload_by_canonical_id, provenance_by_canonical_id).
    """
    from mrta.retrieval.fusion import canonical_identity

    grouped: dict[str, list] = {}
    for record in caption_records:
        candidate = adapter.from_caption_record(record, 0.0, 1)
        cid = "|".join(canonical_identity(candidate))
        grouped.setdefault(cid, []).append(record)

    payloads: dict[str, dict] = {}
    provenance: dict[str, str] = {}
    for cid, records in grouped.items():
        chosen = sorted(
            records,
            key=lambda r: (0 if (r.caption or "").strip() else 1, r.evidence_id),
        )[0]
        payloads[cid] = {
            "figure_caption": chosen.caption,
            "figure_description": chosen.detailed_description,
            "figure_nearby_text": chosen.nearby_text,
        }
        provenance[cid] = (
            "vlm_caption" if (chosen.caption or "").strip() else "nearby_text_fallback"
        )
    return payloads, provenance


# ---------------------------------------------------------------------------
# Frozen baselines (spec 16) — read from artifacts, never hardcoded
# ---------------------------------------------------------------------------


def _derive_at_1_from_per_query(rows: list[dict], retrieved_key: str) -> dict:
    """Recover R@1/H@1/FigR@1 from a frozen per-query artifact's top-5 lists.

    A re-read of frozen data, not a re-run: PR1/PR2 recorded their ranked
    retrieval but only aggregated @5 metrics. Returns {} if the artifact does
    not carry document_id on its retrieved entries (PR3's CLIP list does not),
    in which case those table cells stay N/A rather than being guessed.
    """
    from mrta.eval.retrieval_metrics import figure_recall_at_k, hit_rate_at_k, recall_at_k
    from mrta.eval.types import CanonicalEvidence, RetrievedCandidate

    if not rows or "document_id" not in (rows[0][retrieved_key] or [{}])[0]:
        return {}

    by_intent: dict[str, list] = {}
    r1, h1, f1 = [], [], []
    for row in rows:
        targets = _parse_targets(row["expected_canonical_evidence"])
        cands = [
            RetrievedCandidate(
                candidate_id=e.get("candidate_id", ""),
                evidence=CanonicalEvidence(
                    document_id=e["document_id"],
                    page_number=e["page_number"],
                    figure_id=e.get("figure_id"),
                ),
                score=e.get("score", 0.0),
                rank=i + 1,
            )
            for i, e in enumerate(row[retrieved_key])
        ]
        vals = {
            "recall_at_1": recall_at_k(cands, targets, 1),
            "hit_at_1": hit_rate_at_k(cands, targets, 1),
            "figure_recall_at_1": (
                figure_recall_at_k(cands, targets, 1) if _has_figure_targets(targets) else None
            ),
        }
        by_intent.setdefault(row["intent"], []).append(vals)
        r1.append(vals["recall_at_1"])
        h1.append(vals["hit_at_1"])
        if vals["figure_recall_at_1"] is not None:
            f1.append(vals["figure_recall_at_1"])

    def _mean(v):
        return round(sum(v) / len(v), 4) if v else None

    out = {
        "overall": {
            "recall_at_1": _mean(r1),
            "hit_at_1": _mean(h1),
            "figure_recall_at_1": _mean(f1),
        }
    }
    for intent, vals in by_intent.items():
        out[intent] = {
            "recall_at_1": _mean([v["recall_at_1"] for v in vals]),
            "hit_at_1": _mean([v["hit_at_1"] for v in vals]),
            "figure_recall_at_1": _mean(
                [v["figure_recall_at_1"] for v in vals if v["figure_recall_at_1"] is not None]
            ),
        }
    return out


def _load_frozen_baselines() -> dict:
    """Load PR1-PR4 metrics from their result artifacts (spec 16)."""
    baselines: dict[str, dict] = {}

    pr1 = _load_json(PR1_METRICS_PATH)["metrics"]
    pr1_at1 = _derive_at_1_from_per_query(
        _load_json(PR1_PER_QUERY_PATH), "retrieved_canonical_evidence"
    )
    baselines["PR1 Text"] = _merge_slices(pr1, pr1_at1)

    pr2 = _load_json(PR2_METRICS_PATH)["caption_only"]
    pr2_at1 = _derive_at_1_from_per_query(_load_json(PR2_PER_QUERY_PATH), "caption_retrieved")
    baselines["PR2 Caption"] = _merge_slices(pr2, pr2_at1)

    pr3 = _load_json(PR3_METRICS_PATH)["clip_only"]
    baselines["PR3 CLIP"] = _merge_slices(pr3, {})

    pr4 = _load_json(PR4_METRICS_PATH)
    baselines["PR4 RRF T+C (frozen)"] = {
        "overall": pr4["overall"]["rrf_text_caption"],
        **{i: v["rrf_text_caption"] for i, v in pr4["by_intent"].items()},
    }
    baselines["PR4 RRF T+C+CLIP (frozen)"] = {
        "overall": pr4["overall"]["rrf_text_caption_clip"],
        **{i: v["rrf_text_caption_clip"] for i, v in pr4["by_intent"].items()},
    }
    return baselines


def _merge_slices(base: dict, extra: dict) -> dict:
    out = {}
    for slice_name, metrics in base.items():
        merged = dict(metrics)
        merged.update({k: v for k, v in extra.get(slice_name, {}).items() if v is not None})
        out[slice_name] = merged
    return out


def _pr1_hybrid_both_covered() -> float | None:
    """PR1's both-target hybrid coverage, derived from its frozen per-query rows."""
    from mrta.eval.types import CanonicalEvidence, RetrievedCandidate

    rows = _load_json(PR1_PER_QUERY_PATH)
    flags = []
    for row in rows:
        targets = _parse_targets(row["expected_canonical_evidence"])
        cands = [
            RetrievedCandidate(
                candidate_id="",
                evidence=CanonicalEvidence(
                    document_id=e["document_id"],
                    page_number=e["page_number"],
                    figure_id=e.get("figure_id"),
                ),
                score=0.0,
                rank=i + 1,
            )
            for i, e in enumerate(row["retrieved_canonical_evidence"])
        ]
        flag = _hybrid_both_covered(cands, targets, EVALUATION_K)
        if flag is not None:
            flags.append(flag)
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else None


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
    from mrta.retrieval.reranker import CrossEncoderReranker
    from mrta.retrieval.vector_store import VectorStore

    print("=== PR5 Evaluation — Cross-Encoder Reranking over RRF Fusion (v2) ===\n")
    print(f"candidate_pool_size = {CANDIDATE_POOL_SIZE}")
    print(f"rrf_k               = {RRF_K}")
    print(f"rerank_top_n        = {RERANK_TOP_N}")
    print(f"evaluation_k        = {EVALUATION_K}")
    print(f"reranker model      = {settings.reranker_model_name}")
    print("scores              = RRF and cross-encoder kept separate, never blended\n")

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

    # --- figure text lookup (built once, query-independent) ---
    figure_payloads, figure_provenance = _build_figure_text_lookup(caption_store._records, adapter)
    prov_counts = {
        "vlm_caption": sum(1 for v in figure_provenance.values() if v == "vlm_caption"),
        "nearby_text_fallback": sum(
            1 for v in figure_provenance.values() if v == "nearby_text_fallback"
        ),
    }
    print(
        f"\nfigure text lookup: {len(figure_payloads)} canonical figures "
        f"({prov_counts['vlm_caption']} VLM caption / "
        f"{prov_counts['nearby_text_fallback']} nearby-text fallback)"
    )
    print(
        f"  record-level provenance in the caption index: "
        f"{sum(1 for r in caption_store._records if (r.caption or '').strip())} VLM / "
        f"{sum(1 for r in caption_store._records if not (r.caption or '').strip())} fallback "
        f"of {caption_store.size} records"
    )

    # --- load the cross-encoder (warm-up timed separately from steady state) ---
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    reranker._model.predict([("warm up query", "warm up document")])
    warmup_s = time.perf_counter() - t0
    print(f"\nreranker load : {load_s:.2f}s   first-inference warm-up: {warmup_s:.3f}s")

    # --- PR4 diagnostic query sets (spec 19, 20) ---
    pr4_metrics = _load_json(PR4_METRICS_PATH)
    pr4_rows = {r["query_id"]: r for r in _load_json(PR4_PER_QUERY_PATH)}
    pr4_in_pool_failures: list[str] = pr4_metrics["neither_group_breakdown"][
        "ranking_fusion_opportunity"
    ]
    # Re-derive PR4's push-outs with PR4's own rule: target was top-5 in some
    # single stream but fell below top-5 after three-stream fusion.
    pr4_push_outs: list[str] = []
    for qid, row in pr4_rows.items():
        if not row["has_figure_targets"]:
            continue
        pool = [v for v in row["stream_pool_target_rank"].values() if v is not None]
        if not pool:
            continue
        before, after = min(pool), row["fused_target_rank"]["rrf_text_caption_clip"]
        if after is not None and before <= EVALUATION_K < after:
            pr4_push_outs.append(qid)
    print(
        f"\nPR4 diagnostic sets: {len(pr4_in_pool_failures)} in-pool ranking failures, "
        f"{len(pr4_push_outs)} push-outs "
        f"(PR4 recorded {pr4_metrics['rank_movement']['pushed_out_of_top5']})"
    )

    print(f"\nRunning {len(queries)} queries x {len(ALL_CONFIGS)} configurations ...\n")

    per_query: list[dict] = []
    rerank_latencies: dict[str, list[float]] = {c: [] for c in RERANKED_OF.values()}
    pairs_scored: dict[str, int] = {c: 0 for c in RERANKED_OF.values()}

    for q in queries:
        qid, text, intent = q["query_id"], q["query"], q["intent"]
        targets = _parse_targets(_evidence_list(q))
        has_figs = _has_figure_targets(targets)

        # --- retrieve candidate pools (depth 20, NOT the eval cutoff) ---
        text_hits = text_store.search_with_scores(text, k=CANDIDATE_POOL_SIZE)
        text_cands = [
            adapter.chunk_to_candidate(chunk, score, i + 1)
            for i, (chunk, score) in enumerate(text_hits)
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

        # --- reranker text payloads, carried through fusion ---
        # Text: the actual retrieved chunk text, never page-level text.
        # Figures: the shared production-derived lookup, so caption-only,
        # CLIP-only and merged figures all get the identical representation.
        payloads: dict[str, dict[str, dict]] = {"text": {}, "caption": {}, "clip": {}}
        for (chunk, _), cand in zip(text_hits, text_cands):
            payloads["text"]["|".join(canonical_identity(cand))] = {"chunk_text": chunk.text}
        for stream, cands in (("caption", caption_cands), ("clip", clip_cands)):
            for cand in cands:
                cid = "|".join(canonical_identity(cand))
                if cid in figure_payloads:
                    payloads[stream][cid] = figure_payloads[cid]

        pool_ranks = {
            name: _target_rank(cands, targets, "figure" if has_figs else "any")
            for name, cands in stream_pool.items()
        }

        # Provenance of the target figure's own representation (spec 8, M).
        target_sources = sorted(
            {
                figure_provenance[cid]
                for t in targets
                if t.figure_id is not None
                and (cid := f"figure|{t.document_id}|p{t.page_number}:{t.figure_id}")
                in figure_provenance
            }
        )

        row: dict = {
            "query_id": qid,
            "query": text,
            "intent": intent,
            "paper": q.get("document_id"),
            "retrieval_challenge": q.get("retrieval_challenge"),
            "difficulty": q.get("difficulty"),
            "has_figure_targets": has_figs,
            "target_figure_text_source": target_sources[0] if len(target_sources) == 1 else None,
            "in_pr4_in_pool_failures": qid in pr4_in_pool_failures,
            "in_pr4_push_outs": qid in pr4_push_outs,
            "expected_canonical_evidence": [
                {
                    "document_id": t.document_id,
                    "page_number": t.page_number,
                    "figure_id": t.figure_id,
                }
                for t in targets
            ],
            "stream_pool_target_rank": pool_ranks,
            "configs": {},
            "hybrid_both_covered": {},
            "rrf_target_rank": {},
            "rerank_target_rank": {},
            "rank_movement": {},
            "final_top5": {},
        }

        for rrf_config, stream_names in RRF_CONFIGURATIONS.items():
            streams = {name: stream_pool[name] for name in stream_names}
            fused = reciprocal_rank_fusion_canonical(
                streams, k=RRF_K, top_k=None, payloads=payloads
            )
            fused_scored = [
                adapter.from_fused_candidate(fc, rank=i + 1) for i, fc in enumerate(fused)
            ]

            # RRF baseline: metrics on the top-5 slice (see _metrics_for).
            row["configs"][rrf_config] = _metrics_for(
                fused_scored[:EVALUATION_K], targets, has_figs
            )
            row["hybrid_both_covered"][rrf_config] = _hybrid_both_covered(
                fused_scored[:EVALUATION_K], targets, EVALUATION_K
            )
            row["rrf_target_rank"][rrf_config] = {
                "any": _target_rank(fused_scored, targets, "any"),
                "text": _target_rank(fused_scored, targets, "text"),
                "figure": _target_rank(fused_scored, targets, "figure"),
            }
            row["final_top5"][rrf_config] = [
                {
                    "canonical_id": fc.canonical_id,
                    "evidence_type": fc.evidence_type,
                    "rrf_score": round(fc.score, 6),
                    "rrf_rank": i + 1,
                    "modality_sources": list(fc.modality_sources),
                    "source_ranks": fc.source_ranks,
                    "figure_id": fc.figure_id,
                }
                for i, fc in enumerate(fused[:EVALUATION_K])
            ]

            # --- reranking over the depth-20 fused pool ---
            rerank_config = RERANKED_OF[rrf_config]
            pool = fused[:RERANK_TOP_N]
            t0 = time.perf_counter()
            # top_k = len(pool): the FULL reranked ordering is needed for rank
            # diagnostics. Metrics still use only the top-EVALUATION_K slice.
            reranked = reranker.rerank(text, pool, top_k=len(pool))
            rerank_latencies[rerank_config].append(time.perf_counter() - t0)
            pairs_scored[rerank_config] += len(pool)

            reranked_scored = [
                adapter.from_fused_candidate(rc.candidate, rank=i + 1)
                for i, rc in enumerate(reranked)
            ]
            row["configs"][rerank_config] = _metrics_for(
                reranked_scored[:EVALUATION_K], targets, has_figs
            )
            row["hybrid_both_covered"][rerank_config] = _hybrid_both_covered(
                reranked_scored[:EVALUATION_K], targets, EVALUATION_K
            )
            row["rerank_target_rank"][rerank_config] = {
                "any": _target_rank(reranked_scored, targets, "any"),
                "text": _target_rank(reranked_scored, targets, "text"),
                "figure": _target_rank(reranked_scored, targets, "figure"),
            }
            # Rank movement is only defined inside the depth-20 pool the
            # reranker actually saw; a target below it is unreachable by PR5.
            pool_scored = fused_scored[:RERANK_TOP_N]
            row["rank_movement"][rerank_config] = {
                kind: {
                    "before": _target_rank(pool_scored, targets, kind),
                    "after": _target_rank(reranked_scored, targets, kind),
                }
                for kind in ("any", "text", "figure")
            }
            row["final_top5"][rerank_config] = [
                {
                    "canonical_id": rc.candidate.canonical_id,
                    "evidence_type": rc.candidate.evidence_type,
                    "figure_id": rc.candidate.figure_id,
                    "modality_sources": list(rc.candidate.modality_sources),
                    "source_ranks": rc.candidate.source_ranks,
                    "rrf_score": round(rc.original_rrf_score, 6),
                    "rrf_rank": rc.original_rrf_rank,
                    "reranker_score": round(rc.reranker_score, 6),
                    "reranker_rank": rc.reranker_rank,
                    "reranker_text_source": rc.reranker_text_source,
                    "reranker_text_chars": len(rc.reranker_text),
                }
                for rc in reranked[:EVALUATION_K]
            ]

        per_query.append(row)

        main_cfg = row["configs"]["rerank_text_caption_clip"]
        base_cfg = row["configs"]["rrf_text_caption_clip"]
        mark = "✓" if main_cfg["hit_at_5"] > 0 else "✗"
        delta = main_cfg["mrr"] - base_cfg["mrr"]
        print(
            f"  {qid} [{intent:14s}] {mark} "
            f"R@5={main_cfg['recall_at_5']:.2f} MRR={main_cfg['mrr']:.2f} "
            f"(ΔMRR vs RRF {delta:+.2f}) | {text[:34]!r}"
        )

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------
    intents = sorted({r["intent"] for r in per_query})
    papers = sorted({r["paper"] for r in per_query if r["paper"]})
    challenges = sorted({r["retrieval_challenge"] for r in per_query if r["retrieval_challenge"]})
    sources = sorted(
        {r["target_figure_text_source"] for r in per_query if r["target_figure_text_source"]}
    )

    overall = {c: _aggregate(per_query, c) for c in ALL_CONFIGS}
    by_intent = {
        i: {c: _aggregate([r for r in per_query if r["intent"] == i], c) for c in ALL_CONFIGS}
        for i in intents
    }
    by_paper = {
        p: {c: _aggregate([r for r in per_query if r["paper"] == p], c) for c in ALL_CONFIGS}
        for p in papers
    }
    by_challenge = {
        ch: {
            c: _aggregate([r for r in per_query if r["retrieval_challenge"] == ch], c)
            for c in ALL_CONFIGS
        }
        for ch in challenges
    }
    by_text_source = {
        s: {
            c: _aggregate([r for r in per_query if r["target_figure_text_source"] == s], c)
            for c in ALL_CONFIGS
        }
        for s in sources
    }

    # Recomputed RRF must reproduce the frozen PR4 numbers exactly — a guard
    # against the PR5 pipeline having perturbed fusion rather than only added
    # a stage after it.
    pr4_check = {}
    for cfg in RRF_CONFIGURATIONS:
        frozen = pr4_metrics["overall"][cfg]
        for key in ("recall_at_5", "mrr", "figure_recall_at_5"):
            pr4_check[f"{cfg}.{key}"] = {
                "frozen": frozen[key],
                "recomputed": overall[cfg][key],
                "match": frozen[key] == overall[cfg][key],
            }
    mismatches = [k for k, v in pr4_check.items() if not v["match"]]
    print("\n\n=== PR4 reproduction check (recomputed RRF vs frozen artifact) ===")
    for k, v in pr4_check.items():
        flag = "OK " if v["match"] else "MISMATCH"
        print(f"  {flag} {k:44s} frozen={v['frozen']}  recomputed={v['recomputed']}")
    if mismatches:
        print(f"  WARNING: {len(mismatches)} mismatch(es) — PR5 fusion is not PR4 fusion.")

    # -----------------------------------------------------------------
    # Required comparison table (spec 26)
    # -----------------------------------------------------------------
    frozen_baselines = _load_frozen_baselines()
    table_rows: dict[str, dict] = dict(frozen_baselines)
    table_rows["RRF T+C"] = {
        "overall": overall["rrf_text_caption"],
        **{i: by_intent[i]["rrf_text_caption"] for i in intents},
    }
    table_rows["RRF T+C+CLIP"] = {
        "overall": overall["rrf_text_caption_clip"],
        **{i: by_intent[i]["rrf_text_caption_clip"] for i in intents},
    }
    table_rows["RRF T+C -> CE"] = {
        "overall": overall["rerank_text_caption"],
        **{i: by_intent[i]["rerank_text_caption"] for i in intents},
    }
    table_rows["RRF T+C+CLIP -> CE"] = {
        "overall": overall["rerank_text_caption_clip"],
        **{i: by_intent[i]["rerank_text_caption_clip"] for i in intents},
    }

    def _print_comparison(slice_name: str, title: str) -> None:
        print(f"\n{title}")
        print(
            f"  {'configuration':26s} {'R@1':>7} {'R@5':>7} {'H@1':>7} {'H@5':>7} "
            f"{'MRR@5':>7} {'nDCG@5':>7} {'FigR@1':>7} {'FigR@5':>7}"
        )
        for name, slices in table_rows.items():
            m = slices.get(slice_name)
            if not m:
                print(f"  {name:26s} {'(not recorded in artifact)':>60s}")
                continue
            print(
                f"  {name:26s} {_fmt(m.get('recall_at_1'),7)} {_fmt(m.get('recall_at_5'),7)} "
                f"{_fmt(m.get('hit_at_1'),7)} {_fmt(m.get('hit_at_5'),7)} "
                f"{_fmt(m.get('mrr'),7)} {_fmt(m.get('ndcg_at_5'),7)} "
                f"{_fmt(m.get('figure_recall_at_1'),7)} {_fmt(m.get('figure_recall_at_5'),7)}"
            )

    print("\n\n=== PR1-PR5 comparison (spec 26) ===")
    _print_comparison("overall", "--- Overall (n=100) ---")
    for i in intents:
        n = by_intent[i]["rrf_text_caption"]["sample_count"]
        _print_comparison(i, f"--- {i} (n={n}) ---")

    print("\n\n=== Per-paper (Recall@5 / MRR@5) ===")
    print(f"  {'paper':28s} " + " ".join(f"{c:>24s}" for c in ALL_CONFIGS))
    for p in papers:
        cells = " ".join(
            f"{by_paper[p][c]['recall_at_5']:11.4f}/{by_paper[p][c]['mrr']:<12.4f}"
            for c in ALL_CONFIGS
        )
        print(f"  {p:28s} {cells}")

    # -----------------------------------------------------------------
    # CLIP marginal value after reranking (spec 27)
    # -----------------------------------------------------------------
    print("\n\n=== CLIP marginal value AFTER reranking:  T+C->CE  vs  T+C+CLIP->CE ===")
    print(f"  {'metric / slice':34s} {'T+C->CE':>10} {'T+C+CLIP->CE':>13} {'Δ CLIP':>10}")
    clip_marginal: dict[str, dict] = {}

    def _delta_row(label: str, slice_agg: dict, key: str) -> None:
        b = slice_agg["rerank_text_caption"][key]
        e = slice_agg["rerank_text_caption_clip"][key]
        d = None if (b is None or e is None) else round(e - b, 4)
        clip_marginal[label] = {"text_caption_ce": b, "text_caption_clip_ce": e, "delta": d}
        if d is None:
            print(f"  {label:34s} {_fmt(b,10)} {_fmt(e,13)} {'N/A':>10}")
        else:
            print(f"  {label:34s} {b:10.4f} {e:13.4f} {d:+10.4f}")

    for key, lbl in (
        ("mrr", "Overall MRR@5"),
        ("recall_at_5", "Overall Recall@5"),
        ("figure_recall_at_5", "Overall Figure Recall@5"),
    ):
        _delta_row(lbl, overall, key)
    for i in intents:
        _delta_row(f"{i} MRR@5", by_intent[i], "mrr")
    for i in intents:
        _delta_row(f"{i} FigR@5", by_intent[i], "figure_recall_at_5")

    # -----------------------------------------------------------------
    # The 20 PR4 in-pool ranking failures (spec 19)
    # -----------------------------------------------------------------
    print("\n\n=== PR4's 20 in-pool ranking failures under reranking (spec 19) ===")
    failure_analysis: dict[str, dict] = {}
    for cfg in RERANKED_OF.values():
        rows = [r for r in per_query if r["in_pr4_in_pool_failures"]]
        addressable, unreachable = [], []
        for r in rows:
            before = r["rank_movement"][cfg]["figure"]["before"]
            (addressable if before is not None else unreachable).append(r)
        befores = [r["rank_movement"][cfg]["figure"]["before"] for r in addressable]
        afters = [r["rank_movement"][cfg]["figure"]["after"] for r in addressable]
        into5 = [
            r["query_id"]
            for r, a in zip(addressable, afters)
            if a is not None and a <= EVALUATION_K
        ]
        into1 = [r["query_id"] for r, a in zip(addressable, afters) if a == 1]
        failure_analysis[cfg] = {
            "n_total": len(rows),
            "n_addressable_within_depth_20": len(addressable),
            "n_unreachable_below_depth_20": len(unreachable),
            "unreachable_query_ids": [r["query_id"] for r in unreachable],
            "mean_target_rank_before": round(sum(befores) / len(befores), 4) if befores else None,
            "mean_target_rank_after": (
                round(
                    sum(a for a in afters if a is not None)
                    / len([a for a in afters if a is not None]),
                    4,
                )
                if any(a is not None for a in afters)
                else None
            ),
            "promoted_into_top5": len(into5),
            "promoted_into_top5_query_ids": into5,
            "promoted_into_top1": len(into1),
            "promoted_into_top1_query_ids": into1,
            "pr4_recovered_into_top5": len(
                pr4_metrics["neither_group_breakdown"]["recovered_into_top5_by_fusion"]
            ),
        }
        fa = failure_analysis[cfg]
        print(f"\n  [{cfg}]")
        print(
            f"    of {fa['n_total']} failures, "
            f"{fa['n_addressable_within_depth_20']} are inside the depth-20 rerank pool"
        )
        print(
            f"    {fa['n_unreachable_below_depth_20']} sit below it and are "
            f"structurally unreachable: {fa['unreachable_query_ids']}"
        )
        print(
            f"    mean target rank before -> after : "
            f"{fa['mean_target_rank_before']} -> {fa['mean_target_rank_after']}"
        )
        print(
            f"    promoted into top-5 : {fa['promoted_into_top5']} "
            f"{fa['promoted_into_top5_query_ids']}"
        )
        print(
            f"    promoted into top-1 : {fa['promoted_into_top1']} "
            f"{fa['promoted_into_top1_query_ids']}"
        )
        print(f"    (PR4 RRF recovered {fa['pr4_recovered_into_top5']}/{fa['n_total']})")

    # -----------------------------------------------------------------
    # The 15 PR4 push-outs (spec 20)
    # -----------------------------------------------------------------
    print("\n\n=== PR4's push-outs under reranking (spec 20) ===")
    pushout_analysis: dict[str, dict] = {}
    for cfg in RERANKED_OF.values():
        rows = [r for r in per_query if r["in_pr4_push_outs"]]
        restored, still_below, pushed_lower, unreachable = [], [], [], []
        for r in rows:
            mv = r["rank_movement"][cfg]["figure"]
            before, after = mv["before"], mv["after"]
            if before is None:
                unreachable.append(r["query_id"])
            elif after is not None and after <= EVALUATION_K:
                restored.append(r["query_id"])
            elif after is not None and after > before:
                pushed_lower.append(r["query_id"])
            else:
                still_below.append(r["query_id"])
        pushout_analysis[cfg] = {
            "n_total": len(rows),
            "restored_to_top5": len(restored),
            "restored_query_ids": restored,
            "still_below_top5": len(still_below),
            "still_below_query_ids": still_below,
            "pushed_even_lower": len(pushed_lower),
            "pushed_lower_query_ids": pushed_lower,
            "unreachable_below_depth_20": len(unreachable),
        }
        pa = pushout_analysis[cfg]
        print(f"\n  [{cfg}] n={pa['n_total']}")
        print(f"    restored to top-5 : {pa['restored_to_top5']} {pa['restored_query_ids']}")
        print(f"    still below top-5 : {pa['still_below_top5']} {pa['still_below_query_ids']}")
        print(f"    pushed even lower : {pa['pushed_even_lower']} {pa['pushed_lower_query_ids']}")

    # -----------------------------------------------------------------
    # Rank movement, split by evidence kind (spec 21)
    # -----------------------------------------------------------------
    print("\n\n=== Rank movement inside the depth-20 pool (spec 21) ===")
    rank_movement: dict[str, dict] = {}
    for cfg in RERANKED_OF.values():
        rank_movement[cfg] = {}
        print(f"\n  [{cfg}]")
        print(
            f"    {'evidence':16s} {'n':>4} {'mean before':>12} {'mean after':>11} "
            f"{'median d':>9} {'->top5':>6} {'->top1':>6} {'out of top5':>12}"
        )
        for kind, label, subset in (
            ("text", "text evidence", per_query),
            ("figure", "figure evidence", per_query),
            ("any", "hybrid queries", [r for r in per_query if r["intent"] == "hybrid"]),
        ):
            befores, afters, deltas = [], [], []
            into5 = into1 = out5 = 0
            for r in subset:
                mv = r["rank_movement"][cfg][kind]
                before, after = mv["before"], mv["after"]
                if before is None or after is None:
                    continue
                befores.append(before)
                afters.append(after)
                deltas.append(after - before)
                if before > EVALUATION_K >= after:
                    into5 += 1
                if before > 1 == after:
                    into1 += 1
                if before <= EVALUATION_K < after:
                    out5 += 1
            entry = {
                "n": len(befores),
                "mean_target_rank_before": (
                    round(sum(befores) / len(befores), 4) if befores else None
                ),
                "mean_target_rank_after": round(sum(afters) / len(afters), 4) if afters else None,
                "median_rank_delta": statistics.median(deltas) if deltas else None,
                "promoted_into_top5": into5,
                "promoted_into_top1": into1,
                "pushed_out_of_top5": out5,
            }
            rank_movement[cfg][label] = entry
            print(
                f"    {label:16s} {entry['n']:>4} {_fmt(entry['mean_target_rank_before'],12)} "
                f"{_fmt(entry['mean_target_rank_after'],11)} {_fmt(entry['median_rank_delta'],9)} "
                f"{into5:>6} {into1:>6} {out5:>12}"
            )

    # -----------------------------------------------------------------
    # Hybrid both-target coverage (spec 22)
    # -----------------------------------------------------------------
    print("\n\n=== Hybrid both-target coverage @5 (spec 22) ===")
    hybrid_cov = {"PR1 Text (derived)": _pr1_hybrid_both_covered()}
    for cfg in ALL_CONFIGS:
        hybrid_cov[cfg] = by_intent.get("hybrid", {}).get(cfg, {}).get("both_target_recall_at_5")
    for name, val in hybrid_cov.items():
        print(f"  {name:28s} {_fmt(val)}")
    print("  (fraction of hybrid queries with BOTH required text and figure evidence in top-5)")

    # -----------------------------------------------------------------
    # Caption provenance (spec 8, M)
    # -----------------------------------------------------------------
    print("\n\n=== Figure Recall@5 by target figure's text provenance (spec M) ===")
    print(f"  {'provenance':24s} {'n':>4} " + " ".join(f"{c:>26s}" for c in ALL_CONFIGS))
    for s in sources:
        n = by_text_source[s]["rrf_text_caption"]["sample_count"]
        cells = " ".join(_fmt(by_text_source[s][c]["figure_recall_at_5"], 26) for c in ALL_CONFIGS)
        print(f"  {s:24s} {n:>4} {cells}")

    # -----------------------------------------------------------------
    # Latency (spec 28)
    # -----------------------------------------------------------------
    try:
        import torch

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch_version = torch.__version__
    except Exception:
        device, torch_version = "unknown", "unknown"
    try:
        import sentence_transformers

        st_version = sentence_transformers.__version__
    except Exception:
        st_version = "unknown"

    print("\n\n=== Reranking latency (spec 28) ===")
    print(f"  device={device}  torch={torch_version}  sentence-transformers={st_version}")
    print(
        f"  model load {load_s:.2f}s (excluded below)   "
        f"first-inference warm-up {warmup_s:.3f}s (excluded below)"
    )
    print(
        f"  {'configuration':26s} {'queries':>8} {'pairs':>7} {'total s':>9} "
        f"{'mean ms':>9} {'p50 ms':>8} {'p95 ms':>8} {'pairs/s':>9}"
    )
    latency: dict[str, dict] = {}
    for cfg, samples in rerank_latencies.items():
        ordered = sorted(samples)
        total = sum(samples)
        p50 = statistics.median(ordered)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        latency[cfg] = {
            "queries": len(samples),
            "candidate_pairs_reranked": pairs_scored[cfg],
            "total_rerank_seconds": round(total, 4),
            "mean_ms_per_query": round(1000 * total / len(samples), 3),
            "p50_ms": round(1000 * p50, 3),
            "p95_ms": round(1000 * p95, 3),
            "pairs_per_second": round(pairs_scored[cfg] / total, 2) if total else None,
        }
        m = latency[cfg]
        print(
            f"  {cfg:26s} {m['queries']:>8} {m['candidate_pairs_reranked']:>7} "
            f"{m['total_rerank_seconds']:>9.3f} {m['mean_ms_per_query']:>9.2f} "
            f"{m['p50_ms']:>8.2f} {m['p95_ms']:>8.2f} {m['pairs_per_second']:>9.1f}"
        )
    print("  Local single-process measurement on a warm model. Not a production SLO.")

    # -----------------------------------------------------------------
    # Persist (spec 25) — new files, never overwriting PR1-PR4 artifacts
    # -----------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "status": "measured",
        "benchmark": "v2",
        "dataset_version": dataset["dataset_version"],
        "corpus_version": dataset["corpus_version"],
        "reranker_model": settings.reranker_model_name,
        "reranker_model_kind": (
            "TEXT cross-encoder. It cannot inspect image pixels. Figure candidates "
            "are reranked through their production-derived textual representation "
            "only, so this is a query-aware textual reranker over multimodal "
            "retrieval candidates — NOT an image-text multimodal reranker."
        ),
        "library_versions": {
            "sentence_transformers": st_version,
            "torch": torch_version,
            "python": platform.python_version(),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_device": device,
        },
        "rrf_k": RRF_K,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "rerank_top_n": RERANK_TOP_N,
        "evaluation_k": EVALUATION_K,
        "parameter_note": (
            "No parameter was tuned against v2. candidate_pool_size, rrf_k and "
            "evaluation_k are held at the PR4 values so PR5 remains directly "
            "comparable with the frozen baselines."
        ),
        "score_provenance": (
            "Retrieval cosines, RRF scores and cross-encoder scores are kept "
            "strictly separate. FusedCandidate.score remains the RRF score; the "
            "cross-encoder score is stored on RerankedCandidate. The two are "
            "never blended — the cross-encoder is the final ranking stage and RRF "
            "only orders the candidate set entering it."
        ),
        "candidate_representation_policy": {
            "text": "the retrieved chunk's own text (Chunk.text), never page-level text",
            "figure": (
                "production-derived figure text from the frozen PR2 caption index, "
                "combined under explicit labels in priority order: "
                "'Figure caption: {VLM caption or extracted caption}', "
                "'Description: {VLM detailed_description}', "
                "'Context: {nearby-text fallback}'"
            ),
            "clip_figure": (
                "VisualRecord carries no text, so CLIP-retrieved figures are "
                "represented by the same canonical-id lookup as caption-retrieved "
                "figures. Verified total: the caption and CLIP indices cover the "
                "identical 21 canonical figures, so no figure candidate is textless."
            ),
            "multi_crop_selection": (
                "A canonical figure owning several caption records resolves "
                "deterministically: prefer a record with a VLM caption, then the "
                "lowest evidence_id."
            ),
            "leakage_control": (
                "Candidate text reaches the model only via FusedCandidate.payload, "
                "and only production-derived keys are ever written there. Benchmark "
                "fields (source_note, expected_evidence, retrieval_challenge, "
                "difficulty, canonical figure names) are never written to a payload "
                "and therefore cannot enter model input."
            ),
        },
        "caption_provenance": {
            "canonical_figures": len(figure_payloads),
            "canonical_level": prov_counts,
            "record_level": {
                "vlm_caption": sum(1 for r in caption_store._records if (r.caption or "").strip()),
                "nearby_text_fallback": sum(
                    1 for r in caption_store._records if not (r.caption or "").strip()
                ),
                "total_records": caption_store.size,
            },
            "note": (
                "The 17/20 VLM/fallback split quoted in the PR5 spec is a RECORD-level "
                "count over 37 caption records. Reranking operates on canonical "
                "figures, where the split is different because four multi-crop figures "
                "own both captioned and uncaptioned crops."
            ),
        },
        "configurations": {
            "rrf_text_caption": ["text", "caption"],
            "rrf_text_caption_clip": ["text", "caption", "clip"],
            "rerank_text_caption": ["text", "caption", "-> cross-encoder"],
            "rerank_text_caption_clip": ["text", "caption", "clip", "-> cross-encoder"],
        },
        "metric_depth_note": (
            f"All metrics are computed on the top-{EVALUATION_K} slice, matching the "
            "PR1-PR4 protocol. mean_reciprocal_rank takes no k argument, so the "
            f"reported MRR is MRR@{EVALUATION_K}. The full depth-{RERANK_TOP_N} "
            "reranked ordering is retained only for rank diagnostics."
        ),
        "pr4_reproduction_check": pr4_check,
        "overall": overall,
        "by_intent": by_intent,
        "by_paper": by_paper,
        "by_retrieval_challenge": by_challenge,
        "by_target_figure_text_source": by_text_source,
        "frozen_baselines": frozen_baselines,
        "frozen_baseline_sources": {
            "PR1 Text": str(PR1_METRICS_PATH.relative_to(REPO_ROOT)),
            "PR2 Caption": str(PR2_METRICS_PATH.relative_to(REPO_ROOT)),
            "PR3 CLIP": str(PR3_METRICS_PATH.relative_to(REPO_ROOT)),
            "PR4 RRF": str(PR4_METRICS_PATH.relative_to(REPO_ROOT)),
            "note": (
                "Loaded from result files, never hardcoded. R@1/H@1/FigR@1 for PR1 "
                "and PR2 are re-read from their frozen per-query artifacts, which "
                "recorded ranked retrieval but aggregated only @5 metrics. PR3's "
                "CLIP per-query list carries no document_id, so its @1 cells that "
                "were not already recorded are reported as N/A rather than guessed."
            ),
        },
        "clip_marginal_value_after_reranking": clip_marginal,
        "pr4_in_pool_failure_recovery": failure_analysis,
        "pr4_push_out_recovery": pushout_analysis,
        "rank_movement": rank_movement,
        "hybrid_both_target_coverage_at_5": hybrid_cov,
        "latency": {
            "model_load_seconds": round(load_s, 4),
            "first_inference_warmup_seconds": round(warmup_s, 4),
            "steady_state": latency,
            "note": (
                "Local single-process measurement on a warm model, excluding load "
                "and warm-up. Not a production SLO claim."
            ),
        },
        "limitations": [
            f"Small visual universe: {len(figure_payloads)} canonical figures against a "
            f"depth-{CANDIDATE_POOL_SIZE} candidate pool, so visual-stream agreement is "
            "unusually dense. Carried forward from PR4; not tuned around.",
            "16/21 canonical figures use a page-render fallback rather than a true "
            "figure crop. Figure extraction was not changed in PR5.",
            "The cross-encoder is text-only and never sees image pixels, so failure on "
            "visually specific queries may reflect the textual representation rather "
            "than the reranker's ranking quality.",
            "The nearby-text fallback is low-quality page-extraction boilerplate for "
            "many figures, which bounds what any text reranker can achieve on the "
            "figures that have no VLM caption.",
            "Some PR4 in-pool failures sit below the depth-20 rerank pool and are "
            "structurally unreachable by PR5 at the fixed candidate depth.",
        ],
    }

    metrics_path = RESULTS_DIR / "pr5_reranker_metrics.json"
    per_query_path = RESULTS_DIR / "pr5_per_query.json"
    for path in (metrics_path, per_query_path):
        if path.exists():
            print(f"\nNOTE: overwriting previous PR5 artifact {path.name}")
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query, indent=2), encoding="utf-8")

    print(f"\nSaved → {metrics_path}")
    print(f"Saved → {per_query_path}")


if __name__ == "__main__":
    main()
