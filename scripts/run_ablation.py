"""PR7 ablation runner — executes the frozen ablation matrix over a benchmark.

Usage:
    python scripts/run_ablation.py --config configs/ablation_config.yaml
    python scripts/run_ablation.py --configuration full_reranked --limit 10
    python scripts/run_ablation.py --generation --limit 20

Retrieval-only is the default, so a standard run needs no generator and no
Ollama. Generation is opt-in via --generation.

Requirements for a full v2 run:
    - Ollama with nomic-embed-text (text + caption query embedding)
    - data/vector_store/v2_corpus/            (build_text_index.py --benchmark v2)
    - data/eval/indices/v2/caption_index/     (build_caption_index.py --benchmark v2)
    - data/eval/indices/v2/clip_image_index/  (build_clip_image_index.py --benchmark v2)
    - reranked configurations additionally download the cross-encoder on first use
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BENCHMARK_PATHS = {
    "v1": {
        "queries": REPO_ROOT / "data" / "eval" / "queries_v1.json",
        "vector_store": REPO_ROOT / "data" / "vector_store" / "eval_corpus",
        "caption_index": REPO_ROOT / "data" / "eval" / "indices" / "caption_index",
        "clip_index": REPO_ROOT / "data" / "eval" / "indices" / "clip_image_index",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "manifest.json",
    },
    "v2": {
        "queries": REPO_ROOT / "data" / "eval" / "queries_v2.json",
        "vector_store": REPO_ROOT / "data" / "vector_store" / "v2_corpus",
        "caption_index": REPO_ROOT / "data" / "eval" / "indices" / "v2" / "caption_index",
        "clip_index": REPO_ROOT / "data" / "eval" / "indices" / "v2" / "clip_image_index",
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json",
    },
}


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_stores(benchmark: str, need: set[str]):
    """Load only the indices the selected configurations actually require."""
    from mrta.core.config import settings
    from mrta.eval.ablation_runner import RunnerStores
    from mrta.eval.adapter import EvalAdapter

    paths = BENCHMARK_PATHS[benchmark]
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    # Benchmark targets use manifest document ids; only EvalAdapter can resolve
    # them from a retrieved chunk's source filename.
    stores = RunnerStores(adapter=EvalAdapter(manifest))
    models: dict[str, str | None] = {
        "embedding_model": None,
        "clip_model": None,
        "reranker_model": None,
        "generator_model": None,
    }

    # CLIP first: ImageStore.load() warms torch before FAISS, avoiding the macOS
    # libomp conflict documented in CLIPEmbedder.warmup().
    if "clip" in need:
        from mrta.retrieval.clip_embedder import CLIPEmbedder
        from mrta.retrieval.image_store import ImageStore

        clip = CLIPEmbedder()
        stores.clip = ImageStore.load(paths["clip_index"], clip)
        models["clip_model"] = clip.model_name

    if {"text", "caption"} & need:
        from mrta.retrieval.embedder import Embedder

        embedder = Embedder(settings.embedding_model)
        models["embedding_model"] = settings.embedding_model

        if "text" in need:
            from mrta.retrieval.vector_store import VectorStore

            stores.text = VectorStore.load(paths["vector_store"], embedder)
        if "caption" in need:
            from mrta.retrieval.caption_store import CaptionVectorStore

            stores.caption = CaptionVectorStore.load(paths["caption_index"], embedder)

    return stores, models


def main() -> None:
    from mrta.eval.ablation import (
        CONFIGURATIONS_BY_ID,
        FROZEN_CONFIGURATIONS,
        HISTORICAL_CONFIGURATIONS,
        aggregate,
        percentile,
    )
    from mrta.eval.ablation_runner import AblationRunner

    parser = argparse.ArgumentParser(description="Run the PR7 ablation matrix.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/ablation_config.yaml")
    parser.add_argument("--benchmark", choices=("v1", "v2"), default=None)
    parser.add_argument("--configuration", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--intent", action="append", default=None)
    parser.add_argument("--document-id", action="append", default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help="PR8 generation condition(s): g0_baseline, g1_explicit_citations, "
        "g2_structured_evidence, g3_structured_generation. Default: g0_baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config.exists() else {}
    benchmark = args.benchmark or cfg.get("benchmark", "v2")
    candidate_depth = int(cfg.get("candidate_depth", 20))
    rrf_k = int(cfg.get("rrf_k", 60))
    final_top_k = int(cfg.get("final_top_k", 5))
    output_dir = args.output_dir or REPO_ROOT / cfg.get("output_dir", f"results/{benchmark}/pr7")

    selected_ids = args.configuration or cfg.get(
        "configurations", [c.config_id for c in FROZEN_CONFIGURATIONS]
    )
    unknown = [c for c in selected_ids if c not in CONFIGURATIONS_BY_ID]
    if unknown:
        print(f"ERROR: unknown configuration(s): {unknown}")
        print(f"Known: {sorted(CONFIGURATIONS_BY_ID)}")
        sys.exit(1)
    configs = [CONFIGURATIONS_BY_ID[c] for c in selected_ids]

    run_generation = args.generation and not args.retrieval_only
    generation_ids = set(cfg.get("generation_configurations", []))

    paths = BENCHMARK_PATHS[benchmark]
    dataset = json.loads(paths["queries"].read_text(encoding="utf-8"))
    queries = dataset["queries"]

    if args.query_id:
        queries = [q for q in queries if q["query_id"] in set(args.query_id)]
    if args.intent:
        queries = [q for q in queries if q.get("intent") in set(args.intent)]
    if args.document_id:
        queries = [q for q in queries if q.get("document_id") in set(args.document_id)]
    if args.limit:
        queries = queries[: args.limit]

    # Stable ordering so repeated runs produce byte-comparable artifacts.
    queries = sorted(queries, key=lambda q: q["query_id"])

    needed_streams = {s for c in configs for s in c.streams}
    print(f"=== PR7 Ablation — benchmark {benchmark} ===\n")
    print(f"configurations : {len(configs)}")
    print(f"queries        : {len(queries)}")
    print(f"streams needed : {sorted(needed_streams) or 'none (oracle only)'}")
    print(f"generation     : {'on' if run_generation else 'off (retrieval-only)'}")
    print(f"candidate_depth={candidate_depth}  rrf_k={rrf_k}  final_top_k={final_top_k}\n")

    for name, path in (
        ("text vector store", paths["vector_store"] if "text" in needed_streams else None),
        ("caption index", paths["caption_index"] if "caption" in needed_streams else None),
        ("CLIP index", paths["clip_index"] if "clip" in needed_streams else None),
    ):
        if path is not None and not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    stores, models = build_stores(benchmark, needed_streams)

    reranker = None
    if any(c.rerank for c in configs):
        from mrta.retrieval.reranker import CrossEncoderReranker

        print("Loading cross-encoder ...")
        reranker = CrossEncoderReranker()
        models["reranker_model"] = reranker.model_name

    generator = None
    if run_generation:
        from mrta.core.config import settings
        from mrta.core.llm import LLMClient

        temperature = float(cfg.get("generation", {}).get("temperature", 0.0))

        class _DeterministicGenerator:
            """LLMClient wrapper pinned to the evaluation temperature."""

            def __init__(self) -> None:
                self._llm = LLMClient()

            def generate(self, prompt: str, images: list) -> str:
                return self._llm.chat(
                    [{"role": "user", "content": prompt}], temperature=temperature
                )

        generator = _DeterministicGenerator()
        models["generator_model"] = settings.ollama_llm_model

    from mrta.eval.generation_conditions import CONDITIONS_BY_ID, G0_BASELINE

    condition_ids = args.condition or cfg.get("generation_conditions", [G0_BASELINE])
    unknown_conditions = [c for c in condition_ids if c not in CONDITIONS_BY_ID]
    if unknown_conditions:
        print(f"ERROR: unknown condition(s): {unknown_conditions}")
        print(f"Known: {sorted(CONDITIONS_BY_ID)}")
        sys.exit(1)
    conditions = [CONDITIONS_BY_ID[c] for c in condition_ids]
    if run_generation and len(conditions) > 1:
        print(f"generation conditions: {', '.join(condition_ids)}\n")

    # One runner per condition so the prompt/parsing strategy is fixed per run,
    # while retrieval stays shared: pools are retrieved once per query below and
    # handed to every condition unchanged.
    runners = {
        c.condition_id: AblationRunner(
            stores,
            reranker=reranker,
            generator=generator,
            condition=c,
            candidate_depth=candidate_depth,
            rrf_k=rrf_k,
            final_top_k=final_top_k,
        )
        for c in conditions
    }
    runner = runners[conditions[0].condition_id]

    rows = []
    started = time.perf_counter()

    for index, query in enumerate(queries, start=1):
        pools, stream_latency = runner.retrieve_streams(query["query"])
        # Retrieval ran once; every other condition reuses exactly that evidence.
        for other in runners.values():
            if other is not runner:
                other.adopt_retrieval_state(runner)
        for config in configs:
            generate_here = run_generation and (
                not generation_ids or config.config_id in generation_ids
            )
            # Every generation condition receives the identical pools object.
            active = conditions if generate_here else conditions[:1]
            for condition in active:
                result = runners[condition.condition_id].run_configuration(
                    config, query, pools, stream_latency, generate=generate_here
                )
                rows.append(result)
        marker = "✓" if all(r.status == "ok" for r in rows[-len(configs) :]) else "✗"
        print(f"  [{index:3d}/{len(queries)}] {query['query_id']} {marker}")

    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    by_config = defaultdict(list)
    for row in rows:
        by_config[row.config_id].append(row)

    overall = {cid: aggregate(rs) for cid, rs in by_config.items()}

    def slice_by(attr: str) -> dict:
        values = sorted({getattr(r, attr) for r in rows if getattr(r, attr)})
        return {
            value: {
                cid: aggregate([r for r in rs if getattr(r, attr) == value])
                for cid, rs in by_config.items()
            }
            for value in values
        }

    # PR8: conditions are a second axis. Aggregating by config alone would
    # collapse every generation condition into one bucket and make the
    # comparison invisible.
    generation_rows = [r for r in rows if r.generation_condition]
    condition_ids_present = sorted({r.generation_condition for r in generation_rows})
    config_ids_present = sorted({r.config_id for r in generation_rows})

    # Nested by configuration, never flattened across it. The oracle supplies
    # ground-truth evidence, so averaging it together with a retrieval
    # configuration inflates every condition — exactly what the oracle's own
    # caveat forbids.
    by_condition_per_config = {
        config_id: {
            cid: aggregate(
                [
                    r
                    for r in generation_rows
                    if r.generation_condition == cid and r.config_id == config_id
                ]
            )
            for cid in condition_ids_present
        }
        for config_id in config_ids_present
    }
    primary_config = (
        ("full_reranked" if "full_reranked" in by_condition_per_config else config_ids_present[0])
        if config_ids_present
        else None
    )
    by_condition = by_condition_per_config.get(primary_config, {})
    primary_rows = [r for r in generation_rows if r.config_id == primary_config]

    by_condition_intent = {
        intent: {
            cid: aggregate(
                [r for r in primary_rows if r.generation_condition == cid and r.intent == intent]
            )
            for cid in condition_ids_present
        }
        for intent in sorted({r.intent for r in primary_rows if r.intent})
    }
    by_condition_provenance = {
        prov: {
            cid: aggregate(
                [
                    r
                    for r in primary_rows
                    if r.generation_condition == cid and r.target_figure_provenance == prov
                ]
            )
            for cid in condition_ids_present
        }
        for prov in sorted(
            {r.target_figure_provenance for r in primary_rows if r.target_figure_provenance}
        )
    }

    by_intent = slice_by("intent")
    by_paper = slice_by("document_id")
    by_challenge = slice_by("retrieval_challenge")
    # Two different populations, deliberately reported separately.
    by_target_provenance = slice_by("target_figure_provenance")
    by_retrieved_provenance = slice_by("top_retrieved_figure_provenance")

    latency_percentiles = {
        cid: {
            stage: {
                "p50": percentile([r.latency_ms.get(stage) for r in rs], 50),
                "p95": percentile([r.latency_ms.get(stage) for r in rs], 95),
            }
            for stage in ("retrieval", "fusion", "rerank", "generation", "total")
        }
        for cid, rs in by_config.items()
    }

    metadata = {
        "benchmark": benchmark,
        "dataset_version": dataset.get("dataset_version"),
        "corpus_version": dataset.get("corpus_version"),
        "query_count": len(queries),
        "configuration_count": len(configs),
        "candidate_depth": candidate_depth,
        "rrf_k": rrf_k,
        "final_top_k": final_top_k,
        "generation_enabled": run_generation,
        "generation_temperature": (
            float(cfg.get("generation", {}).get("temperature", 0.0)) if run_generation else None
        ),
        "models": models,
        "generation_conditions": [
            {
                "condition_id": c.condition_id,
                "template": c.template,
                "structured_output": c.structured_output,
                "prompt_hash": c.prompt_hash(),
                "description": c.description,
            }
            for c in conditions
        ],
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_seconds": round(elapsed, 2),
        "historical_configurations": list(HISTORICAL_CONFIGURATIONS),
        "metric_notes": {
            "mrr": (
                f"MRR is computed on the top-{final_top_k} slice (MRR@{final_top_k}); "
                "mean_reciprocal_rank takes no k, so scoring the full pool would "
                f"silently yield MRR@{candidate_depth}."
            ),
            "support_proxy": (
                "lexical_support_score and supported_claim_fraction are deterministic "
                "lexical proxies, not semantic faithfulness measurements."
            ),
            "oracle": (
                "oracle_evidence_generation receives ground-truth evidence directly; "
                "its retrieval metrics are perfect by construction and must never be "
                "compared against retrieval configurations."
            ),
        },
    }

    summary = {
        "metadata": metadata,
        "overall": overall,
        "by_intent": by_intent,
        "by_paper": by_paper,
        "by_retrieval_challenge": by_challenge,
        "by_target_figure_provenance": by_target_provenance,
        "by_top_retrieved_figure_provenance": by_retrieved_provenance,
        "latency_percentiles": latency_percentiles,
        "primary_generation_config": primary_config,
        "by_generation_condition": by_condition,
        "by_generation_condition_per_config": by_condition_per_config,
        "by_generation_condition_intent": by_condition_intent,
        "by_generation_condition_target_provenance": by_condition_provenance,
        "generation_condition_paired": _paired_analysis(primary_rows),
        "generation_condition_paired_oracle": (
            _paired_analysis(
                [r for r in generation_rows if r.config_id == "oracle_evidence_generation"]
            )
            if "oracle_evidence_generation" in config_ids_present
            else {}
        ),
        "evidence_hash_integrity": _evidence_hash_integrity(generation_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "ablation_per_query.json").write_text(
        json.dumps([r.as_dict() for r in rows], indent=2), encoding="utf-8"
    )
    (output_dir / "ablation_config_resolved.yaml").write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "candidate_depth": candidate_depth,
                "rrf_k": rrf_k,
                "final_top_k": final_top_k,
                "configurations": [c.config_id for c in configs],
                "generation_enabled": run_generation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_tables(overall, configs)
    if by_condition:
        print("\n=== PR8 generation conditions (overall) ===")
        print(
            f"  {'condition':28s} {'CitP':>8} {'CitR':>8} {'CitF1':>8} "
            f"{'FigCov':>8} {'HybBoth':>8} {'ClaimCov':>9} {'CtxTok':>8}"
        )
        for cid, m in by_condition.items():
            print(
                f"  {cid:28s} {_fmt(m.get('citation_precision'))} "
                f"{_fmt(m.get('citation_recall'))} {_fmt(m.get('citation_f1'))} "
                f"{_fmt(m.get('figure_target_covered'))} "
                f"{_fmt(m.get('both_targets_covered'))} "
                f"{_fmt(m.get('claim_citation_coverage'),9)} "
                f"{_fmt(m.get('context_token_count'))}"
            )
        integrity = summary["evidence_hash_integrity"]
        print(
            f"\n  evidence-hash integrity: "
            f"{integrity['pairs_with_identical_evidence']}/{integrity['pairs_checked']} "
            f"(query, configuration) pairs identical across conditions "
            f"({'PASS' if integrity['all_identical'] else 'FAIL'})"
        )
        for config_id, counts in sorted(integrity["per_configuration"].items()):
            print(
                f"    {config_id:30s} identical={counts['identical']:3d} "
                f"divergent={counts['divergent']:3d}"
            )
    _write_report(output_dir, summary, configs, by_intent, by_target_provenance)

    print(f"\nSaved → {output_dir}/ablation_summary.json")
    print(f"Saved → {output_dir}/ablation_per_query.json")
    print(f"Saved → {output_dir}/ablation_report.md")
    print(f"\nRows: {len(rows)}  ({len(queries)} queries x {len(configs)} configurations)")
    print(f"Elapsed: {elapsed:.1f}s")


def _paired_analysis(rows) -> dict:
    """G3-vs-G0 win/tie/loss on the primary metrics, paired per query.

    Paired because every condition answers the same queries from the same
    evidence: a mean delta can hide a change that helps a few queries a lot
    while hurting many slightly.
    """
    from mrta.eval.generation_conditions import (
        G0_BASELINE,
        G1_EXPLICIT_CITATIONS,
        G2_STRUCTURED_EVIDENCE,
        G3_STRUCTURED_GENERATION,
        paired_comparison,
    )

    configs = {r.config_id for r in rows}
    if len(configs) > 1:
        raise ValueError(
            f"paired analysis requires one configuration, got {sorted(configs)}: "
            "pairing across configurations compares different evidence"
        )
    present = {r.generation_condition for r in rows}
    out: dict = {}
    metrics = ("citation_recall", "citation_f1", "citation_precision", "claim_citation_coverage")

    for baseline_id, treatment_id in (
        (G0_BASELINE, G1_EXPLICIT_CITATIONS),
        (G1_EXPLICIT_CITATIONS, G2_STRUCTURED_EVIDENCE),
        (G2_STRUCTURED_EVIDENCE, G3_STRUCTURED_GENERATION),
        (G0_BASELINE, G3_STRUCTURED_GENERATION),
    ):
        if baseline_id not in present or treatment_id not in present:
            continue
        base_by_query = {r.query_id: r for r in rows if r.generation_condition == baseline_id}
        treat_by_query = {r.query_id: r for r in rows if r.generation_condition == treatment_id}
        shared = sorted(set(base_by_query) & set(treat_by_query))
        pair_key = f"{treatment_id}_vs_{baseline_id}"
        out[pair_key] = {
            metric: paired_comparison(
                [base_by_query[q].generation_metrics.get(metric) for q in shared],
                [treat_by_query[q].generation_metrics.get(metric) for q in shared],
            )
            for metric in metrics
        }
    return out


def _evidence_hash_integrity(rows) -> dict:
    """Whether every generation condition saw identical evidence per query.

    PR8's causal claim is only valid where this holds, so it is reported as a
    result rather than assumed.
    """
    from collections import defaultdict

    # Keyed by (query, configuration), not query alone: different retrieval
    # configurations legitimately supply different evidence, and the oracle
    # supplies ground truth by design. The invariant PR8 needs is that the
    # generation *conditions* agree within one configuration.
    hashes: dict[tuple[str, str], set] = defaultdict(set)
    for row in rows:
        if row.evidence_context_hash:
            hashes[(row.query_id, row.config_id)].add(row.evidence_context_hash)

    mismatched = sorted(key for key, h in hashes.items() if len(h) > 1)
    per_configuration: dict[str, dict] = defaultdict(lambda: {"identical": 0, "divergent": 0})
    for (_, config_id), h in hashes.items():
        per_configuration[config_id]["identical" if len(h) == 1 else "divergent"] += 1

    return {
        "grouping": "(query_id, config_id)",
        "pairs_checked": len(hashes),
        "pairs_with_identical_evidence": len(hashes) - len(mismatched),
        "mismatched": [{"query_id": q, "config_id": c} for q, c in mismatched],
        "per_configuration": dict(per_configuration),
        "all_identical": not mismatched,
    }


def _fmt(value, width: int = 8) -> str:
    return f"{value:{width}.4f}" if isinstance(value, (int, float)) else f"{'N/A':>{width}}"


def _print_tables(overall: dict, configs) -> None:
    print("\n=== Retrieval (overall) ===")
    header = (
        f"  {'configuration':28s} {'R@1':>8} {'R@5':>8} {'MRR@5':>8} {'nDCG@5':>8} {'FigR@5':>8}"
    )
    print(header)
    for config in configs:
        m = overall.get(config.config_id, {})
        print(
            f"  {config.config_id:28s} {_fmt(m.get('recall_at_1'))} {_fmt(m.get('recall_at_5'))} "
            f"{_fmt(m.get('mrr_at_5'))} {_fmt(m.get('ndcg_at_5'))} "
            f"{_fmt(m.get('figure_recall_at_5'))}"
        )

    generated = [c for c in configs if overall.get(c.config_id, {}).get("generated_count")]
    if generated:
        print("\n=== Generation (overall) ===")
        print(
            f"  {'configuration':28s} {'CitP':>8} {'CitR':>8} {'CitF1':>8} "
            f"{'Valid':>8} {'Support':>8} {'CtxTok':>8}"
        )
        for config in generated:
            m = overall[config.config_id]
            print(
                f"  {config.config_id:28s} {_fmt(m.get('citation_precision'))} "
                f"{_fmt(m.get('citation_recall'))} {_fmt(m.get('citation_f1'))} "
                f"{_fmt(m.get('citation_validity_rate'))} "
                f"{_fmt(m.get('lexical_support_score'))} "
                f"{_fmt(m.get('context_token_count'))}"
            )


def _md_row(values) -> str:
    return "| " + " | ".join(values) + " |"


def _md_num(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"


def _write_report(output_dir: Path, summary: dict, configs, by_intent, by_provenance) -> None:
    meta = summary["metadata"]
    overall = summary["overall"]
    lines: list[str] = []

    lines.append("# PR7 — Ablation Report")
    lines.append("")
    lines.append(
        f"Benchmark **{meta['benchmark']}** ({meta['dataset_version']}), "
        f"{meta['query_count']} queries x {meta['configuration_count']} configurations. "
        f"Generated {meta['timestamp_utc']}."
    )
    lines.append("")
    lines.append(
        f"`candidate_depth={meta['candidate_depth']}` · `rrf_k={meta['rrf_k']}` · "
        f"`final_top_k={meta['final_top_k']}` · commit `{(meta.get('git_commit') or 'n/a')[:10]}`"
    )
    lines.append("")
    lines.append("Models: " + ", ".join(f"{k}={v}" for k, v in meta["models"].items() if v))
    lines.append("")

    lines.append("> **Support metrics are deterministic lexical proxies.** They are useful for")
    lines.append("> regression testing, but they underestimate support for correct paraphrases")
    lines.append("> and must not be read as a complete semantic faithfulness metric.")
    lines.append("")
    lines.append("> **`oracle_evidence_generation` is evaluation-only.** It receives ground-truth")
    lines.append("> evidence directly, so its retrieval metrics are perfect by construction and")
    lines.append("> are never comparable with a retrieval configuration.")
    lines.append("")

    lines.append("## Retrieval comparison")
    lines.append("")
    lines.append(_md_row(["Configuration", "R@1", "R@5", "MRR@5", "nDCG@5", "FigR@1", "FigR@5"]))
    lines.append(_md_row(["---"] * 7))
    for config in configs:
        m = overall.get(config.config_id, {})
        lines.append(
            _md_row(
                [
                    f"`{config.config_id}`",
                    _md_num(m.get("recall_at_1")),
                    _md_num(m.get("recall_at_5")),
                    _md_num(m.get("mrr_at_5")),
                    _md_num(m.get("ndcg_at_5")),
                    _md_num(m.get("figure_recall_at_1")),
                    _md_num(m.get("figure_recall_at_5")),
                ]
            )
        )
    lines.append("")

    generated = [c for c in configs if overall.get(c.config_id, {}).get("generated_count")]
    lines.append("## Generation comparison")
    lines.append("")
    if not generated:
        lines.append("_Generation was not run for this execution (retrieval-only mode)._")
    else:
        lines.append(
            _md_row(
                [
                    "Configuration",
                    "Citation P",
                    "Citation R",
                    "Citation F1",
                    "Validity",
                    "Support proxy",
                    "Context tokens",
                ]
            )
        )
        lines.append(_md_row(["---"] * 7))
        for config in generated:
            m = overall[config.config_id]
            lines.append(
                _md_row(
                    [
                        f"`{config.config_id}`",
                        _md_num(m.get("citation_precision")),
                        _md_num(m.get("citation_recall")),
                        _md_num(m.get("citation_f1")),
                        _md_num(m.get("citation_validity_rate")),
                        _md_num(m.get("lexical_support_score")),
                        _md_num(m.get("context_token_count")),
                    ]
                )
            )
    lines.append("")

    lines.append("## Latency (mean ms)")
    lines.append("")
    lines.append(_md_row(["Configuration", "Retrieval", "Fusion", "Rerank", "Generation", "Total"]))
    lines.append(_md_row(["---"] * 6))
    for config in configs:
        m = overall.get(config.config_id, {})
        lines.append(
            _md_row(
                [
                    f"`{config.config_id}`",
                    _md_num(m.get("latency_retrieval_ms")),
                    _md_num(m.get("latency_fusion_ms")),
                    _md_num(m.get("latency_rerank_ms")),
                    _md_num(m.get("latency_generation_ms")),
                    _md_num(m.get("latency_total_ms")),
                ]
            )
        )
    lines.append("")

    for title, sliced in (("Intent", by_intent), ("Target figure provenance", by_provenance)):
        lines.append(f"## By {title.lower()}")
        lines.append("")
        for value, per_config in sliced.items():
            n = next(iter(per_config.values())).get("sample_count", 0)
            lines.append(f"### {value} (n={n})")
            lines.append("")
            lines.append(_md_row(["Configuration", "R@5", "MRR@5", "FigR@5"]))
            lines.append(_md_row(["---"] * 4))
            for config in configs:
                m = per_config.get(config.config_id, {})
                lines.append(
                    _md_row(
                        [
                            f"`{config.config_id}`",
                            _md_num(m.get("recall_at_5")),
                            _md_num(m.get("mrr_at_5")),
                            _md_num(m.get("figure_recall_at_5")),
                        ]
                    )
                )
            lines.append("")

    condition_summary = summary.get("by_generation_condition") or {}
    if condition_summary:
        lines.append("## PR8 generation conditions")
        lines.append("")
        integrity = summary["evidence_hash_integrity"]
        primary = summary.get("primary_generation_config", "unknown")
        lines.append(
            f"Comparison configuration: **`{primary}`**. These numbers are for that "
            "configuration alone — the oracle is reported separately below and is never "
            "averaged in, because it is handed ground-truth evidence and scores near "
            "perfectly by construction."
        )
        lines.append("")
        lines.append(
            f"Evidence-hash integrity: "
            f"**{integrity['pairs_with_identical_evidence']}/{integrity['pairs_checked']}** "
            f"(query, configuration) pairs supplied identical evidence to every condition "
            f"({'PASS' if integrity['all_identical'] else 'FAIL'}). "
            "Retrieval is frozen; only the prompt and output contract vary."
        )
        lines.append("")
        lines.append(
            "> **G3 is a combined intervention.** It changes two things relative to G2 — "
            "typed evidence cards *and* the JSON `{answer, citations}` contract — so its "
            "improvement demonstrates the effect of the combination, **not** of JSON output "
            "alone. Separating them would need a 'G0 prompt + JSON output' cell, which is "
            "absent from the frozen matrix; adding one after seeing results would be tuning "
            "rather than measurement."
        )
        lines.append("")
        lines.append(
            _md_row(
                [
                    "Condition",
                    "Cit P",
                    "Cit R",
                    "Cit F1",
                    "Validity",
                    "Fig coverage",
                    "Hybrid both",
                    "Claim citation cov",
                ]
            )
        )
        lines.append(_md_row(["---"] * 8))
        for cid, m in condition_summary.items():
            lines.append(
                _md_row(
                    [
                        f"`{cid}`",
                        _md_num(m.get("citation_precision")),
                        _md_num(m.get("citation_recall")),
                        _md_num(m.get("citation_f1")),
                        _md_num(m.get("citation_validity_rate")),
                        _md_num(m.get("figure_target_covered")),
                        _md_num(m.get("both_targets_covered")),
                        _md_num(m.get("claim_citation_coverage")),
                    ]
                )
            )
        lines.append("")

        lines.append("### Context cost")
        lines.append("")
        lines.append(
            _md_row(["Condition", "Context tokens", "Answer tokens", "Generation mean ms"])
        )
        lines.append(_md_row(["---"] * 4))
        for cid, m in condition_summary.items():
            lines.append(
                _md_row(
                    [
                        f"`{cid}`",
                        _md_num(m.get("context_token_count")),
                        _md_num(m.get("answer_token_count")),
                        _md_num(m.get("latency_generation_ms")),
                    ]
                )
            )
        lines.append("")

        paired = summary.get("generation_condition_paired") or {}
        if paired:
            lines.append("### Paired comparison (per-query, same evidence)")
            lines.append("")
            lines.append(
                _md_row(
                    ["Comparison", "Metric", "Treatment better", "Tie", "Baseline better", "Mean Δ"]
                )
            )
            lines.append(_md_row(["---"] * 6))
            for pair_key, metrics in paired.items():
                for metric, counts in metrics.items():
                    lines.append(
                        _md_row(
                            [
                                f"`{pair_key}`",
                                metric,
                                str(counts["treatment_better"]),
                                str(counts["tie"]),
                                str(counts["baseline_better"]),
                                _md_num(counts["mean_delta"]),
                            ]
                        )
                    )
            lines.append("")

    per_config = summary.get("by_generation_condition_per_config") or {}
    oracle_summary = per_config.get("oracle_evidence_generation")
    if oracle_summary:
        lines.append("### Oracle condition (diagnostic only)")
        lines.append("")
        lines.append(
            "The oracle receives the benchmark's ground-truth evidence directly, so its "
            "retrieval metrics are perfect by construction. It bounds how completely the "
            "generator cites evidence when availability is not the constraint. It is **not** "
            "comparable with a retrieval configuration and must never be averaged with one."
        )
        lines.append("")
        lines.append(_md_row(["Condition", "Cit R", "Cit F1", "Claim citation cov"]))
        lines.append(_md_row(["---"] * 4))
        for cid, m in oracle_summary.items():
            lines.append(
                _md_row(
                    [
                        f"`{cid}`",
                        _md_num(m.get("citation_recall")),
                        _md_num(m.get("citation_f1")),
                        _md_num(m.get("claim_citation_coverage")),
                    ]
                )
            )
        lines.append("")

    lines.append("## Failure decomposition")
    lines.append("")
    lines.append(_md_row(["Configuration", "Successful", "Failed", "Failure categories"]))
    lines.append(_md_row(["---"] * 4))
    for config in configs:
        m = overall.get(config.config_id, {})
        categories = m.get("failure_categories", {})
        rendered = ", ".join(f"{k}={v}" for k, v in categories.items() if k != "none") or "—"
        lines.append(
            _md_row(
                [
                    f"`{config.config_id}`",
                    str(m.get("successful_count", 0)),
                    str(m.get("failed_count", 0)),
                    rendered,
                ]
            )
        )
    lines.append("")

    (output_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
