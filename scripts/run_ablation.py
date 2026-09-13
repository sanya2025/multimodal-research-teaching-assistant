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

    runner = AblationRunner(
        stores,
        reranker=reranker,
        generator=generator,
        candidate_depth=candidate_depth,
        rrf_k=rrf_k,
        final_top_k=final_top_k,
    )

    rows = []
    started = time.perf_counter()

    for index, query in enumerate(queries, start=1):
        pools, stream_latency = runner.retrieve_streams(query["query"])
        for config in configs:
            generate_here = run_generation and (
                not generation_ids or config.config_id in generation_ids
            )
            result = runner.run_configuration(
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
    _write_report(output_dir, summary, configs, by_intent, by_target_provenance)

    print(f"\nSaved → {output_dir}/ablation_summary.json")
    print(f"Saved → {output_dir}/ablation_per_query.json")
    print(f"Saved → {output_dir}/ablation_report.md")
    print(f"\nRows: {len(rows)}  ({len(queries)} queries x {len(configs)} configurations)")
    print(f"Elapsed: {elapsed:.1f}s")


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
