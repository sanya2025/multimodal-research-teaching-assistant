"""Build a text VectorStore for a versioned evaluation corpus.

v1's text index (data/vector_store/aiayn/) was built ad hoc in the Phase-03
notebook and is not rebuilt by this script — it is loaded as-is by
run_eval_baseline.py. This script exists because v2 has no equivalent
pre-built store: it is multi-document (5 papers), so each document is loaded,
chunked (recursive strategy, chunker.py defaults), and added to one shared
VectorStore, exactly mirroring the notebook recipe that produced aiayn but
generalized to loop over multiple PDFs.

Usage:
    python scripts/build_text_index.py --benchmark v2

Requirements:
    - Ollama running with nomic-embed-text (used to embed chunks)
    - [pdf] extra installed (PyMuPDF)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BENCHMARKS = {
    "v2": {
        "manifest": REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json",
        "papers_dir": REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "papers",
        "output": REPO_ROOT / "data" / "vector_store" / "v2_corpus",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    args = parser.parse_args()
    cfg = BENCHMARKS[args.benchmark]

    from mrta.ingestion.chunker import chunk_pdf
    from mrta.ingestion.pdf_loader import load_pdf
    from mrta.retrieval.embedder import Embedder
    from mrta.retrieval.vector_store import VectorStore

    print(f"=== Build Text Index ({args.benchmark}) ===\n")

    manifest = json.loads(cfg["manifest"].read_text(encoding="utf-8"))
    embedder = Embedder("nomic-embed-text")
    store = VectorStore(embedder)

    for doc in manifest["documents"]:
        pdf_path = cfg["papers_dir"] / doc["filename"]
        print(f"Loading {doc['filename']} ...")
        pdf = load_pdf(pdf_path)
        chunks = chunk_pdf(pdf, strategy="recursive")
        # Chunk.source is the filename mrta wrote into the PageRecord — but the
        # manifest's canonical document_id is keyed off the *original* filename
        # (e.g. "clip.pdf"), and load_pdf() sets source=pdf_path.name, so these
        # already match: EvalAdapter's source->document_id lookup will resolve.
        store.add(chunks)
        print(f"  {len(chunks)} chunks added (doc_id={pdf.doc_id})")

    print(f"\nTotal chunks: {len(store._chunks)}")
    cfg["output"].mkdir(parents=True, exist_ok=True)
    store.save(cfg["output"])
    print(f"Saved → {cfg['output']}")


if __name__ == "__main__":
    main()
