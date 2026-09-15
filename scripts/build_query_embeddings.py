"""Freeze the benchmark query embeddings so offline evaluation needs no Ollama.

The v2 indices are keyed to ``nomic-embed-text``, which Embedder serves over the
Ollama REST API. Precomputing the 100 query vectors once lets CI reproduce the
frozen retrieval numbers exactly without running a model server.

Run this only when the benchmark queries or the embedding model change — both of
which invalidate the frozen baselines and need their own justification.

Usage:
    python scripts/build_query_embeddings.py --benchmark v2

Requirements:
    - Ollama running with the model named in the target indices' config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BENCHMARKS = {
    "v1": {
        "queries": REPO_ROOT / "data" / "eval" / "queries_v1.json",
        "index_config": REPO_ROOT / "data" / "eval" / "indices" / "caption_index" / "config.json",
        "output": REPO_ROOT / "data" / "eval" / "query_embeddings_v1.npz",
    },
    "v2": {
        "queries": REPO_ROOT / "data" / "eval" / "queries_v2.json",
        "index_config": (
            REPO_ROOT / "data" / "eval" / "indices" / "v2" / "caption_index" / "config.json"
        ),
        "output": REPO_ROOT / "data" / "eval" / "query_embeddings_v2.npz",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="v2")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the embedding model. Defaults to the index's own config.json, "
        "which is the only value that can produce comparable vectors.",
    )
    args = parser.parse_args()

    paths = BENCHMARKS[args.benchmark]
    queries_path = Path(paths["queries"])
    output_path = Path(paths["output"])

    # The index decides the model, not settings/.env — a mismatch here silently
    # produces vectors that cannot be compared against the stored index.
    index_config = json.loads(Path(paths["index_config"]).read_text(encoding="utf-8"))
    model = args.model or index_config["model"]
    expected_dim = index_config.get("dim")

    data = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = [q["query"] for q in data["queries"]]
    print(f"benchmark={args.benchmark}  queries={len(queries)}  model={model}")

    from mrta.eval.query_embedding_cache import QueryEmbeddingCache
    from mrta.retrieval.embedder import Embedder

    cache = QueryEmbeddingCache.build(
        queries=queries,
        embedder=Embedder(model),
        queries_path=queries_path,
    )

    if expected_dim is not None and cache.dim != expected_dim:
        print(
            f"ERROR: embedded dim {cache.dim} != index dim {expected_dim}. "
            "These vectors could not be searched against the frozen index.",
            file=sys.stderr,
        )
        return 1

    cache.save(output_path)
    size_kb = output_path.stat().st_size / 1024
    print(f"wrote {output_path.relative_to(REPO_ROOT)}  ({len(cache)} vectors, {size_kb:.0f} KB)")
    print(f"  dim={cache.dim}  model={cache.model_name}")
    print(f"  queries_sha256={cache.queries_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
