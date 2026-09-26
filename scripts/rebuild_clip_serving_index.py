"""Rebuild the production CLIP serving index from the caption index.

Why this exists
---------------
``data/vector_store/clip_images/`` is written by ``/upload`` via
``mrta.ingestion.document_indexer.index_document``. Until ADR-014 that write
silently failed: the API handed ``ImageStore`` the ``mrta.multimodal``
CLIPEmbedder, which takes a PIL image and has no ``warmup()``, while
``ImageStore`` embeds by path. The resulting exception was caught by the
per-stream degradation handler, so ingestion reported success and the CLIP index
was simply never created.

Re-uploading every document repairs it, but that re-runs PDF parsing, text
chunking and VLM captioning. This script instead rebuilds only the missing CLIP
index, reusing the figure assets and canonical identities the caption index
already recorded. It needs CLIP weights but no Ollama, so it also works offline.

This is a repair tool, not part of ingestion: new uploads build the index
correctly once the wiring fix is deployed.

Usage:
    python scripts/rebuild_clip_serving_index.py            # rebuild
    python scripts/rebuild_clip_serving_index.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mrta.core.schemas import EvidenceRecord, VisualRecord  # noqa: E402
from mrta.ingestion.document_indexer import (  # noqa: E402
    CAPTION_INDEX_NAME,
    CLIP_INDEX_NAME,
    production_figure_id,
)


def load_caption_records(caption_dir: Path) -> list[EvidenceRecord]:
    """Read the persisted caption index metadata."""
    metadata = caption_dir / "metadata.jsonl"
    if not metadata.exists():
        return []
    return [
        EvidenceRecord.model_validate_json(line)
        for line in metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def to_visual_records(records: list[EvidenceRecord]) -> tuple[list[VisualRecord], list[str]]:
    """Convert figure caption records to CLIP-index records, reporting what was skipped.

    A record is skipped when it is not a figure, carries no image_path, or points
    at a file that no longer exists. Skips are returned rather than logged away:
    a silently shrinking index is exactly the failure this script repairs.
    """
    visual: list[VisualRecord] = []
    skipped: list[str] = []

    for rec in records:
        if rec.modality != "image":
            continue
        if not rec.image_path:
            skipped.append(f"{rec.evidence_id}: no image_path")
            continue
        if not (REPO_ROOT / rec.image_path).exists():
            skipped.append(f"{rec.evidence_id}: missing asset {rec.image_path}")
            continue
        figure_index = rec.figure_index if rec.figure_index is not None else 1
        visual.append(
            VisualRecord(
                document_id=rec.doc_id,
                page=rec.page,
                figure_id=production_figure_id(rec.page, figure_index),
                figure_index=figure_index,
                image_path=rec.image_path,
                extraction_method=rec.extraction_method,
            )
        )
    return visual, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-root",
        type=Path,
        default=REPO_ROOT / "data" / "vector_store",
        help="Vector store root (default: data/vector_store)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be indexed without writing the index",
    )
    args = parser.parse_args()

    caption_dir = args.store_root / CAPTION_INDEX_NAME
    clip_dir = args.store_root / CLIP_INDEX_NAME

    records = load_caption_records(caption_dir)
    if not records:
        print(f"ERROR: no caption index at {caption_dir}")
        print("Upload a document first, or point --store-root elsewhere.")
        raise SystemExit(1)

    visual_records, skipped = to_visual_records(records)
    print(f"Caption records read : {len(records)}")
    print(f"Figures to embed     : {len(visual_records)}")
    for note in skipped:
        print(f"  skipped {note}")

    if not visual_records:
        print("Nothing to index.")
        raise SystemExit(1)

    if args.dry_run:
        print(f"\nDry run — would write {clip_dir}")
        return

    # Imported here so --dry-run does not pay for loading torch.
    from mrta.retrieval.clip_embedder import CLIPEmbedder
    from mrta.retrieval.image_store import ImageStore

    print("\nLoading CLIP encoder...")
    store = ImageStore(CLIPEmbedder())
    store.add_images(visual_records)
    store.save(clip_dir)

    print(f"Wrote {store.size} vectors to {clip_dir}")
    config = clip_dir / "config.json"
    if config.exists():
        print(f"  config: {json.loads(config.read_text(encoding='utf-8'))}")
    print("\nRestart the API so the new index is loaded.")


if __name__ == "__main__":
    main()
