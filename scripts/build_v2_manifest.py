"""Build data/eval/corpus/v2/manifest.json from real figure extraction.

This script implements the v2 figure-grounding policy documented in
notes/Build and Validate MRTA Benchmark v2, Then Rerun PR1-PR3.md section 6:
canonical figure_id values proposed in data/queries_v2_grounded.json are bound
to MRTA's actual extract_figures() output wherever that output genuinely
represents the figure, and to a full-page render (mrta.ingestion.page_renderer,
an existing production capability for vector-graphic content) otherwise.

Why a page-render fallback is needed at all
--------------------------------------------
extract_figures() only captures embedded raster images (its own docstring:
"Vector-only figures are not captured"). Manual inspection of all 21 target
figures across the 5 pinned papers found:

  - 5 figures are genuine raster content and bind directly to extracted crops.
  - 16 figures are vector-drawn (matplotlib line charts, box-and-arrow
    architecture diagrams, pseudocode) or are compound diagrams where the only
    embedded raster is an illustrative stock photo, not the diagram itself
    (e.g. BLIP-2's Figures 1-3 each embed a "sunset" or "cat" stock photo next
    to a vector-drawn box diagram; the photo alone does not represent "the
    figure"). These fall back to a full-page render.

This was verified by visual inspection (rendering candidate pages and viewing
them), not assumed. See the PR notes for the inspection transcript.

Icon-exclusion heuristic
-------------------------
Raw extraction also picks up decorative assets that are not figure content at
all: chat-UI avatar icons (LLaVA), a recurring illustrative stock photo + box
icon reused across multiple figures (BLIP-2), a recurring model/chip icon
(SigLIP). These are excluded from raster binding by two rules, applied to
each (width, height) size class within a document:

  A. Appears on >= 3 distinct pages of the same document (a reused UI asset).
  B. Appears >= 3 times on a single page AND both dimensions < 200px (a sheet
     of small illustrative glyphs embedded inside one vector diagram, e.g.
     CLIP Figure 1's small photo icons).

Manually confirmed exceptions
-------------------------------
BLIP-2's page-1 crop (986x564, "sunset" stock photo) escapes both heuristic
rules — it is large and appears on only one page — but visual inspection
confirmed it is decorative, not the Figure 1 diagram. It is excluded by
explicit override (_MANUAL_EXCLUDE_CROPS), documented here rather than folded
silently into the heuristic.

Usage:
    python scripts/build_v2_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CORPUS_DIR = REPO_ROOT / "data" / "eval" / "corpus" / "v2"
PAPERS_DIR = CORPUS_DIR / "papers"
QUERIES_PATH = REPO_ROOT / "data" / "queries_v2_grounded.json"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
RENDERS_DIR = CORPUS_DIR / "page_renders"

# document_id -> filename, matching the pinned papers/ directory
DOCUMENTS = {
    "attention_is_all_you_need": "attention_is_all_you_need.pdf",
    "clip": "clip.pdf",
    "siglip": "siglip.pdf",
    "blip2": "blip2.pdf",
    "llava": "llava.pdf",
}

# (document_id, page, figure_index) crops that pass the heuristic but were
# manually confirmed (by rendering and viewing) to be decorative, not figure
# content. See module docstring.
_MANUAL_EXCLUDE_CROPS: set[tuple[str, int, int]] = {
    ("blip2", 1, 2),  # 986x564 "sunset" stock photo inside the Fig.1 diagram
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _icon_exclusion_set(images: list[dict]) -> set[tuple[int, int]]:
    """(width, height) size classes that are decorative icons, not figures.

    Rule A: recurs on >= 3 distinct pages (a reused UI asset).
    Rule B: recurs >= 3 times on one page and both dims < 200px (a sheet of
    small illustrative glyphs embedded inside a larger vector diagram).
    """
    pages_by_dim: dict[tuple[int, int], set[int]] = defaultdict(set)
    count_by_dim_page: dict[tuple[tuple[int, int], int], int] = defaultdict(int)
    for img in images:
        dim = (img["width"], img["height"])
        pages_by_dim[dim].add(img["page"])
        count_by_dim_page[(dim, img["page"])] += 1

    excluded = {dim for dim, pages in pages_by_dim.items() if len(pages) >= 3}
    for (dim, _page), count in count_by_dim_page.items():
        if count >= 3 and max(dim) < 200:
            excluded.add(dim)
    return excluded


def _figure_targets_from_queries(queries: list[dict]) -> dict[str, tuple[str, int]]:
    """figure_id -> (document_id, page_number), asserting single-location consistency."""
    locations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for q in queries:
        for ev in q["expected_evidence"]:
            fid = ev.get("figure_id")
            if fid:
                locations[fid].add((ev["document_id"], ev["page_number"]))

    resolved: dict[str, tuple[str, int]] = {}
    for fid, locs in locations.items():
        if len(locs) > 1:
            raise ValueError(f"figure_id {fid!r} referenced at multiple locations: {locs}")
        resolved[fid] = next(iter(locs))
    return resolved


def main() -> None:
    from mrta.ingestion.figure_extractor import extract_figures
    from mrta.ingestion.page_renderer import render_page

    print("=== Build v2 Manifest from Real Extraction ===\n")

    dataset = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    figure_targets = _figure_targets_from_queries(dataset["queries"])
    print(f"Query file names {len(figure_targets)} canonical figure_id targets.\n")

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Per-document extraction ----
    documents_manifest: list[dict] = []
    all_extracted: dict[str, list[dict]] = {}  # doc_id -> raw extracted image dicts

    for doc_id, filename in DOCUMENTS.items():
        pdf_path = PAPERS_DIR / filename
        if not pdf_path.exists():
            print(f"ERROR: pinned PDF missing: {pdf_path}")
            sys.exit(1)

        import fitz  # noqa: PLC0415

        with fitz.open(pdf_path) as fh:
            page_count = fh.page_count

        figs = extract_figures(pdf_path)
        images = [
            {
                "page": f.page,
                "figure_index": f.figure_index,
                "width": f.width,
                "height": f.height,
                "image_bytes": f.image_bytes,
            }
            for f in figs
        ]
        all_extracted[doc_id] = images

        print(
            f"{doc_id:28s} pages={page_count:3d}  extracted_raster={len(images):3d}  "
            f"sha256={_sha256(pdf_path)[:16]}..."
        )

        documents_manifest.append(
            {
                "document_id": doc_id,
                "filename": filename,
                "title": None,  # filled in below if available from PDF metadata
                "sha256": _sha256(pdf_path),
                "pages": page_count,
                "source_note": "Pinned for MRTA v2 evaluation benchmark.",
                "figures": [],  # filled in below
            }
        )

    print()

    # ---- Bind each canonical figure_id to extraction output ----
    images_dir = CORPUS_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc_by_id = {d["document_id"]: d for d in documents_manifest}
    raster_bound, fallback_bound = 0, 0

    for figure_id, (doc_id, page) in sorted(figure_targets.items()):
        images = all_extracted[doc_id]
        exclude = _icon_exclusion_set(images)
        candidates = [
            im
            for im in images
            if im["page"] == page
            and (im["width"], im["height"]) not in exclude
            and (doc_id, im["page"], im["figure_index"]) not in _MANUAL_EXCLUDE_CROPS
        ]

        if candidates:
            for im in candidates:
                fname = f"{doc_id}_p{page}_f{im['figure_index']}.png"
                (images_dir / fname).write_bytes(im["image_bytes"])
                doc_by_id[doc_id]["figures"].append(
                    {
                        "figure_id": figure_id,
                        "page_number": page,
                        "figure_index": im["figure_index"],
                        "extraction_type": "raster_crop",
                        "image_path": str((images_dir / fname).relative_to(REPO_ROOT)),
                        "width": im["width"],
                        "height": im["height"],
                    }
                )
            raster_bound += 1
            print(
                f"  RASTER   {figure_id:32s} {doc_id:26s} p{page:<3d} " f"{len(candidates)} crop(s)"
            )
        else:
            pdf_path = PAPERS_DIR / DOCUMENTS[doc_id]
            rec = render_page(pdf_path, page, dpi=150)
            fname = f"{doc_id}_p{page}_page_render.png"
            out_path = RENDERS_DIR / fname
            out_path.write_bytes(rec.image_bytes)
            doc_by_id[doc_id]["figures"].append(
                {
                    "figure_id": figure_id,
                    "page_number": page,
                    "figure_index": 0,  # convention: 0 = whole-page render, not a raster crop
                    "extraction_type": "page_render_fallback",
                    "image_path": str(out_path.relative_to(REPO_ROOT)),
                    "width": None,
                    "height": None,
                }
            )
            fallback_bound += 1
            print(
                f"  FALLBACK {figure_id:32s} {doc_id:26s} p{page:<3d} "
                "(page render, no raster figure content)"
            )

    print()
    print(f"Raster-bound figures  : {raster_bound}")
    print(f"Page-render fallbacks : {fallback_bound}")
    print(f"Total canonical figures: {raster_bound + fallback_bound}")

    manifest = {
        "corpus_version": "v2.0.0",
        "page_numbering_convention": (
            "1-indexed physical PDF page (identical to MRTA ingestion page field; "
            "same convention as v1)"
        ),
        "grounding_policy": (
            "Figure targets bound to real extract_figures() output where the "
            "extracted raster crop genuinely represents the figure. Where the "
            "figure is vector-drawn (chart, box-and-arrow diagram, pseudocode) or "
            "extraction only captured a decorative photo embedded next to the "
            "diagram, a full-page render (mrta.ingestion.page_renderer) is used "
            "instead, tagged extraction_type='page_render_fallback' with "
            "figure_index=0. See scripts/build_v2_manifest.py module docstring "
            "for the exact heuristic and manual verification notes."
        ),
        "documents": documents_manifest,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
