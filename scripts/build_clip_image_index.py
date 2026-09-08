"""Build the PR3 CLIP visual index for the pinned evaluation corpus.

Pipeline::

    corpus PDF
      → extract_figures()            (same PyMuPDF pipeline PR2 used)
      → manifest lookup              (canonical figure_id per (source, page, index))
      → CLIP image encoder           (openai/clip-vit-base-patch32)
      → ImageStore (IndexFlatIP)     → data/eval/indices/clip_image_index/

Embeddings come from the figure images themselves. No caption, description, or
nearby text is used — that is the entire point of the ablation.

Figures whose (source, page, figure_index) is absent from the evaluation manifest
are skipped: without a canonical figure_id they cannot be scored against the
frozen PR1 ground truth.

Usage:
    python scripts/build_clip_image_index.py

Requirements:
    - [multimodal] extra installed (transformers, torch)
    - [pdf] extra installed (PyMuPDF)
    - data/eval/corpus/v1/papers/attention_is_all_you_need.pdf
    - data/eval/corpus/v1/manifest.json

The CLIP weights are downloaded from HuggingFace on first run (~600 MB, cached).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CORPUS_PDF = (
    REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "papers" / "attention_is_all_you_need.pdf"
)
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json"
INDEX_DIR = REPO_ROOT / "data" / "eval" / "indices" / "clip_image_index"
IMAGES_DIR = INDEX_DIR / "images"


def _build_figure_map(manifest: dict) -> tuple[dict[tuple[str, int, int], str], dict[str, str]]:
    """Return (source, page, figure_index) → figure_id and filename → document_id."""
    figure_map: dict[tuple[str, int, int], str] = {}
    source_to_docid: dict[str, str] = {}
    for doc in manifest["documents"]:
        source_to_docid[doc["filename"]] = doc["document_id"]
        for fig in doc.get("figures", []):
            key = (doc["filename"], fig["page_number"], fig["figure_index"])
            figure_map[key] = fig["figure_id"]
    return figure_map, source_to_docid


def main() -> None:
    from mrta.core.schemas import VisualRecord
    from mrta.ingestion.figure_extractor import extract_figures
    from mrta.retrieval.clip_embedder import CLIP_MODEL_ID, CLIPEmbedder
    from mrta.retrieval.image_store import ImageStore

    print("=== Build CLIP Image Index (PR3) ===")
    print()

    if not CORPUS_PDF.exists():
        print(f"ERROR: Corpus PDF not found at {CORPUS_PDF}")
        sys.exit(1)
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    figure_map, source_to_docid = _build_figure_map(manifest)

    print(f"Model      : {CLIP_MODEL_ID}")
    print(f"Corpus PDF : {CORPUS_PDF.name}")
    print(f"Manifest   : {len(figure_map)} canonical figure entries")
    print()

    print("Extracting figures ...")
    figures = extract_figures(CORPUS_PDF)
    print(f"  Extracted {len(figures)} raster figure(s)")
    print()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[VisualRecord] = []
    skipped: list[str] = []

    for fig in figures:
        key = (fig.source, fig.page, fig.figure_index)
        figure_id = figure_map.get(key)
        if figure_id is None:
            skipped.append(f"p{fig.page}/f{fig.figure_index}")
            continue

        document_id = source_to_docid.get(fig.source, fig.source)
        filename = f"{document_id}_p{fig.page}_f{fig.figure_index}.png"
        image_path = IMAGES_DIR / filename
        image_path.write_bytes(fig.image_bytes)

        records.append(
            VisualRecord(
                document_id=document_id,
                page=fig.page,
                figure_id=figure_id,
                figure_index=fig.figure_index,
                image_path=str(image_path.relative_to(REPO_ROOT)),
            )
        )
        print(
            f"  p{fig.page} f{fig.figure_index}  →  {figure_id:26s}"
            f"  {fig.width}×{fig.height}  {len(fig.image_bytes) / 1024:.0f} KB"
        )

    if skipped:
        print()
        print(f"  Skipped {len(skipped)} figure(s) absent from manifest: {', '.join(skipped)}")

    if not records:
        print()
        print("ERROR: No figures could be mapped to canonical figure IDs.")
        sys.exit(1)

    print()
    print(f"Embedding {len(records)} figure(s) with CLIP (first run downloads weights) ...")
    embedder = CLIPEmbedder()
    store = ImageStore(embedder)
    store.add_images(records)
    print(f"  Indexed {store.size} record(s), dim={embedder.dim}")

    store.save(INDEX_DIR)

    provenance = {
        "built_at": datetime.now(UTC).isoformat(),
        "corpus_pdf": str(CORPUS_PDF.relative_to(REPO_ROOT)),
        "model": CLIP_MODEL_ID,
        "embedding_dimension": embedder.dim,
        "normalization": "L2",
        "similarity": "inner_product/cosine",
        "index_type": "IndexFlatIP",
        "embedding_source": "direct_clip_image_embedding",
        "captions_used": False,
        "image_bytes_persisted": False,
        "n_records": len(records),
        "records": [
            {
                "record_id": r.record_id,
                "document_id": r.document_id,
                "page": r.page,
                "figure_id": r.figure_id,
                "figure_index": r.figure_index,
                "image_path": r.image_path,
            }
            for r in records
        ],
    }
    (INDEX_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print()
    print(f"Saved index      → {INDEX_DIR}")
    print(f"Saved provenance → {INDEX_DIR / 'provenance.json'}")
    print()
    print(
        f"Index holds {store.size} figure(s) across "
        f"{len({r.figure_id for r in records})} canonical figure ID(s)."
    )


if __name__ == "__main__":
    main()
