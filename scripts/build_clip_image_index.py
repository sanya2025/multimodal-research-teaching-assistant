"""Build the PR3 CLIP visual index for a pinned evaluation corpus.

v1 (default) pipeline::

    single corpus PDF
      → extract_figures()            (PyMuPDF raster extraction)
      → manifest lookup              (canonical figure_id per (source, page, index))
      → CLIP image encoder           (openai/clip-vit-base-patch32)
      → ImageStore (IndexFlatIP)     → data/eval/indices/clip_image_index/

v2 pipeline (--benchmark v2)::

    data/eval/corpus/v2/manifest.json  (already grounded by build_v2_manifest.py)
      → for each figure: read image_path directly (raster crop OR page-render
        fallback for vector-drawn figures — see build_v2_manifest.py docstring)
      → CLIP image encoder
      → ImageStore                       → data/eval/indices/v2/clip_image_index/

In both cases, embeddings come from the figure images themselves. No caption,
description, or nearby text is used — that is the entire point of the ablation.
v2's page-render fallback images are still direct image embeddings (of the
rendered page), not text — the ablation property is preserved.

Usage:
    python scripts/build_clip_image_index.py                  # v1 (default, unchanged)
    python scripts/build_clip_image_index.py --benchmark v2    # v2 (5 papers, 37 figures)

Requirements:
    - [multimodal] extra installed (transformers, torch)
    - [pdf] extra installed (PyMuPDF)
    - v1: data/eval/corpus/v1/papers/attention_is_all_you_need.pdf + manifest.json
    - v2: data/eval/corpus/v2/manifest.json (build with build_v2_manifest.py)

The CLIP weights are downloaded from HuggingFace on first run (~600 MB, cached).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- v1 paths (unchanged) ---
CORPUS_PDF = (
    REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "papers" / "attention_is_all_you_need.pdf"
)
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json"
INDEX_DIR = REPO_ROOT / "data" / "eval" / "indices" / "clip_image_index"
IMAGES_DIR = INDEX_DIR / "images"

# --- v2 paths ---
V2_MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json"
V2_INDEX_DIR = REPO_ROOT / "data" / "eval" / "indices" / "v2" / "clip_image_index"


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


def _write_provenance(index_dir: Path, embedder, records, extra: dict) -> None:
    provenance = {
        "built_at": datetime.now(UTC).isoformat(),
        "model": embedder.model_name,
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
        **extra,
    }
    (index_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def _run_v1() -> None:
    from mrta.core.schemas import VisualRecord
    from mrta.ingestion.figure_extractor import extract_figures
    from mrta.retrieval.clip_embedder import CLIP_MODEL_ID, CLIPEmbedder
    from mrta.retrieval.image_store import ImageStore

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
    print(f"Manifest   : {len(figure_map)} canonical figure entries\n")

    print("Extracting figures ...")
    figures = extract_figures(CORPUS_PDF)
    print(f"  Extracted {len(figures)} raster figure(s)\n")

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
        print(f"\n  Skipped {len(skipped)} figure(s) absent from manifest: {', '.join(skipped)}")

    if not records:
        print("\nERROR: No figures could be mapped to canonical figure IDs.")
        sys.exit(1)

    print(f"\nEmbedding {len(records)} figure(s) with CLIP (first run downloads weights) ...")
    embedder = CLIPEmbedder()
    store = ImageStore(embedder)
    store.add_images(records)
    print(f"  Indexed {store.size} record(s), dim={embedder.dim}")

    store.save(INDEX_DIR)
    _write_provenance(
        INDEX_DIR, embedder, records, {"corpus_pdf": str(CORPUS_PDF.relative_to(REPO_ROOT))}
    )

    print(f"\nSaved index      → {INDEX_DIR}")
    print(f"Saved provenance → {INDEX_DIR / 'provenance.json'}")
    print(
        f"\nIndex holds {store.size} figure(s) across "
        f"{len({r.figure_id for r in records})} canonical figure ID(s)."
    )


def _run_v2() -> None:
    from mrta.core.schemas import VisualRecord
    from mrta.retrieval.clip_embedder import CLIP_MODEL_ID, CLIPEmbedder
    from mrta.retrieval.image_store import ImageStore

    if not V2_MANIFEST_PATH.exists():
        print(f"ERROR: v2 manifest not found at {V2_MANIFEST_PATH}")
        print("Build it first: python scripts/build_v2_manifest.py")
        sys.exit(1)

    manifest = json.loads(V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    all_figures = [
        (doc["document_id"], fig) for doc in manifest["documents"] for fig in doc.get("figures", [])
    ]

    print(f"Model    : {CLIP_MODEL_ID}")
    print(
        f"Manifest : {len(all_figures)} grounded figure record(s) across "
        f"{len(manifest['documents'])} document(s)\n"
    )

    records: list[VisualRecord] = []
    for document_id, fig in all_figures:
        img_path = REPO_ROOT / fig["image_path"]
        if not img_path.exists():
            print(f"ERROR: image artifact missing for {fig['figure_id']}: {img_path}")
            sys.exit(1)
        records.append(
            VisualRecord(
                document_id=document_id,
                page=fig["page_number"],
                figure_id=fig["figure_id"],
                figure_index=fig["figure_index"],
                image_path=fig["image_path"],
            )
        )
        print(
            f"  {fig['figure_id']:32s} {document_id:26s} p{fig['page_number']:<3d} "
            f"({fig['extraction_type']})"
        )

    print(f"\nEmbedding {len(records)} figure(s) with CLIP (first run downloads weights) ...")
    embedder = CLIPEmbedder()
    store = ImageStore(embedder)
    store.add_images(records)
    print(f"  Indexed {store.size} record(s), dim={embedder.dim}")

    V2_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(V2_INDEX_DIR)
    _write_provenance(
        V2_INDEX_DIR,
        embedder,
        records,
        {"manifest": str(V2_MANIFEST_PATH.relative_to(REPO_ROOT))},
    )

    print(f"\nSaved index      → {V2_INDEX_DIR}")
    print(f"Saved provenance → {V2_INDEX_DIR / 'provenance.json'}")
    print(
        f"\nIndex holds {store.size} figure(s) across "
        f"{len({r.figure_id for r in records})} canonical figure ID(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()

    print(f"=== Build CLIP Image Index ({args.benchmark}) ===\n")

    if args.benchmark == "v1":
        _run_v1()
    else:
        _run_v2()


if __name__ == "__main__":
    main()
