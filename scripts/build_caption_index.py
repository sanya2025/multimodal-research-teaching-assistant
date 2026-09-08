"""Build a FAISS caption index for a versioned evaluation corpus.

v1 (default): extracts figures directly from the single pinned PDF via
extract_figures(), exactly as before — unchanged behavior.

v2: reads the already-grounded figure list from data/eval/corpus/v2/manifest.json
(built by scripts/build_v2_manifest.py) instead of re-running extraction. This
matters because v2's manifest includes page-render fallbacks for vector-drawn
figures (16 of 21) alongside raster crops (5 of 21) — re-running extract_figures()
here would only see the raster crops and silently drop the fallback figures.
Reading from the manifest keeps the caption index consistent with the grounding
decisions already made and validated there.

In both cases, the retrieval text embedded for each figure is the output of
EvidenceRecord.retrieval_text() — caption → detailed_description → nearby_text —
populated by VisualAnalyzer.analyze_evidence(). This is the same text the
production MRTA pipeline would embed; no manually authored captions are used.

Usage:
    python scripts/build_caption_index.py                  # v1 (default, unchanged)
    python scripts/build_caption_index.py --benchmark v2    # v2 (5 papers, 37 figures)

Requirements:
    - Ollama running with the configured VLM (default: qwen2.5vl:latest)
    - Ollama running with the configured embedding model (default: nomic-embed-text)
    - v1: data/eval/corpus/v1/papers/attention_is_all_you_need.pdf exists
    - v2: data/eval/corpus/v2/manifest.json exists (build with build_v2_manifest.py)

Provenance is recorded in <index_dir>/provenance.json.
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
INDEX_DIR = REPO_ROOT / "data" / "eval" / "indices" / "caption_index"
IMAGES_DIR = INDEX_DIR / "images"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json"

# --- v2 paths ---
V2_MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json"
V2_PAPERS_DIR = REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "papers"
V2_INDEX_DIR = REPO_ROOT / "data" / "eval" / "indices" / "v2" / "caption_index"

# Matches figure_extractor.py's own nearby-text window, so the fallback text
# v2 uses is the same convention v1 relies on (caption > description > nearby).
_NEARBY_TEXT_CHARS = 400


def _probe_ollama(host: str, vlm_model: str, embed_model: str) -> None:
    import httpx

    try:
        r = httpx.get(f"{host}/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {host}: {e}")
        print("Start Ollama with: ollama serve")
        sys.exit(1)

    available = {m["name"] for m in r.json().get("models", [])}
    missing = []
    for model in (vlm_model, embed_model):
        base = model.split(":")[0]
        if not any(base in name for name in available):
            missing.append(model)
    if missing:
        for m in missing:
            print(f"ERROR: Model not available in Ollama: {m}")
            print(f"  Pull it with: ollama pull {m}")
        sys.exit(1)


def _save_figure_image(record, images_dir: Path) -> str:
    """Save the figure's PNG bytes to disk and return the relative path."""
    fname = f"{record.doc_id}_p{record.page}_f{record.figure_index}.png"
    img_path = images_dir / fname
    img_path.write_bytes(record.image_bytes)
    return str(img_path.relative_to(REPO_ROOT))


def _analyze_and_record(ev, analyzer, provenance_entries: list[dict], extra: dict) -> str:
    """Run VisualAnalyzer, compute retrieval_text, append a provenance entry."""
    analyzer.analyze_evidence(ev)

    retrieval_text = ev.retrieval_text()
    fallback_used = False
    if not retrieval_text:
        retrieval_text = ev.nearby_text or ""
        ev.nearby_text = retrieval_text
        fallback_used = True

    caption_source = (
        "vlm_generated"
        if ev.caption or ev.detailed_description
        else ("nearby_text_fallback" if retrieval_text else "empty")
    )

    print(f"    caption  : {ev.caption!r}")
    print(f"    ret_text : {retrieval_text[:80]!r}{'...' if len(retrieval_text) > 80 else ''}")

    provenance_entries.append(
        {
            "evidence_id": ev.evidence_id,
            "doc_id": ev.doc_id,
            "source": ev.source,
            "page": ev.page,
            "figure_index": ev.figure_index,
            "image_path": ev.image_path,
            "caption_source": caption_source,
            "fallback_used": fallback_used,
            "caption": ev.caption,
            "detailed_description": ev.detailed_description,
            "visual_type": ev.visual_type,
            "retrieval_text_chars": len(retrieval_text),
            **extra,
        }
    )
    return retrieval_text


def _run_v1(vlm_model: str, embed_model: str) -> None:
    from mrta.core.schemas import EvidenceRecord
    from mrta.ingestion.figure_extractor import extract_figures
    from mrta.multimodal.visual_analyzer import VisualAnalyzer
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.embedder import Embedder

    if not CORPUS_PDF.exists():
        print(f"ERROR: Corpus PDF not found at {CORPUS_PDF}")
        sys.exit(1)

    print(f"Extracting figures from {CORPUS_PDF.name} ...")
    figure_records = extract_figures(CORPUS_PDF)
    print(f"  Found {len(figure_records)} figure(s)\n")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    analyzer = VisualAnalyzer()
    embedder = Embedder(embed_model)
    store = CaptionVectorStore(embedder)

    evidence_records: list[EvidenceRecord] = []
    provenance_entries: list[dict] = []

    for fig in figure_records:
        print(f"  Analyzing p{fig.page} fig{fig.figure_index} ({fig.width}×{fig.height}) ...")
        ev = fig.to_evidence_record()
        ev.image_path = _save_figure_image(fig, IMAGES_DIR)
        _analyze_and_record(
            ev,
            analyzer,
            provenance_entries,
            {
                "vlm_model": vlm_model,
                "embed_model": embed_model,
                "description_source": "vlm_generated",
            },
        )
        evidence_records.append(ev)

    print(f"\nEmbedding and indexing {len(evidence_records)} record(s) ...")
    store.add(evidence_records)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(INDEX_DIR)

    provenance = {
        "built_at": datetime.now(UTC).isoformat(),
        "corpus_pdf": str(CORPUS_PDF.relative_to(REPO_ROOT)),
        "vlm_model": vlm_model,
        "embed_model": embed_model,
        "description_source": "vlm_generated",
        "retrieval_text_field": (
            "EvidenceRecord.retrieval_text() = caption or detailed_description or nearby_text"
        ),
        "image_bytes_persisted": False,
        "n_records": len(evidence_records),
        "records": provenance_entries,
    }
    prov_path = INDEX_DIR / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Saved caption index → {INDEX_DIR}")
    print(f"Saved provenance    → {prov_path}")
    print(f"\nIndex contains {store.size} record(s) ready for PR2 evaluation.")


def _page_text_excerpt(pdf_path: Path, page_number: int, chars: int = _NEARBY_TEXT_CHARS) -> str:
    """First `chars` characters of a page's text, for use as a nearby_text fallback."""
    import fitz  # noqa: PLC0415

    with fitz.open(pdf_path) as doc:
        text = doc[page_number - 1].get_text("text")
    return text[:chars].strip()


def _run_v2(vlm_model: str, embed_model: str) -> None:
    from mrta.core.schemas import EvidenceRecord
    from mrta.multimodal.visual_analyzer import VisualAnalyzer
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.embedder import Embedder

    if not V2_MANIFEST_PATH.exists():
        print(f"ERROR: v2 manifest not found at {V2_MANIFEST_PATH}")
        print("Build it first: python scripts/build_v2_manifest.py")
        sys.exit(1)

    manifest = json.loads(V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    all_figures = [
        (doc["document_id"], doc["filename"], fig)
        for doc in manifest["documents"]
        for fig in doc.get("figures", [])
    ]
    print(
        f"Manifest names {len(all_figures)} grounded figure record(s) across "
        f"{len(manifest['documents'])} document(s).\n"
    )

    analyzer = VisualAnalyzer()
    embedder = Embedder(embed_model)
    store = CaptionVectorStore(embedder)

    evidence_records: list[EvidenceRecord] = []
    provenance_entries: list[dict] = []
    source_counts: dict[str, int] = {}

    for document_id, filename, fig in all_figures:
        img_path = REPO_ROOT / fig["image_path"]
        if not img_path.exists():
            print(f"  ERROR: image artifact missing for {fig['figure_id']}: {img_path}")
            sys.exit(1)

        image_bytes = img_path.read_bytes()
        eid = f"{document_id}_p{fig['page_number']}_f{fig['figure_index']}_{fig['figure_id']}"
        # nearby_text fallback: same convention as figure_extractor.py — without
        # this, retrieval_text() falls through to "" whenever the VLM caption is
        # empty (which happens for several figures below), and embedding an
        # empty string raises inside Embedder._embed_ollama.
        nearby_text = _page_text_excerpt(V2_PAPERS_DIR / filename, fig["page_number"])
        ev = EvidenceRecord(
            evidence_id=eid,
            doc_id=document_id,
            # source must be the PDF filename, not document_id: EvalAdapter resolves
            # both document_id and figure_id via manifest lookups keyed by filename
            # (doc["filename"]), matching v1's convention (Chunk.source is also a
            # filename). Using document_id here would silently break figure_id
            # resolution for every v2 caption candidate.
            source=filename,
            page=fig["page_number"],
            modality="image",
            # NOT `fig["figure_index"] or None` — page_render_fallback figures use
            # figure_index=0 by convention (see build_v2_manifest.py), and `0 or
            # None` evaluates to None in Python, which would silently break
            # EvalAdapter's figure_id lookup (it requires figure_index is not None).
            figure_index=fig["figure_index"],
            image_bytes=image_bytes,
            image_path=fig["image_path"],
            nearby_text=nearby_text or None,
        )

        print(
            f"  Analyzing {fig['figure_id']} ({document_id} p{fig['page_number']}, "
            f"{fig['extraction_type']}) ..."
        )
        _analyze_and_record(
            ev,
            analyzer,
            provenance_entries,
            {
                "figure_id": fig["figure_id"],
                "extraction_type": fig["extraction_type"],
                "vlm_model": vlm_model,
                "embed_model": embed_model,
            },
        )
        evidence_records.append(ev)
        source_counts[fig["extraction_type"]] = source_counts.get(fig["extraction_type"], 0) + 1

    print(f"\nEmbedding and indexing {len(evidence_records)} record(s) ...")
    store.add(evidence_records)

    V2_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(V2_INDEX_DIR)

    caption_source_counts: dict[str, int] = {}
    for entry in provenance_entries:
        caption_source_counts[entry["caption_source"]] = (
            caption_source_counts.get(entry["caption_source"], 0) + 1
        )

    provenance = {
        "built_at": datetime.now(UTC).isoformat(),
        "manifest": str(V2_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "vlm_model": vlm_model,
        "embed_model": embed_model,
        "retrieval_text_field": (
            "EvidenceRecord.retrieval_text() = caption or detailed_description or nearby_text"
        ),
        "image_bytes_persisted": False,
        "n_records": len(evidence_records),
        "extraction_type_counts": source_counts,
        "caption_source_counts": caption_source_counts,
        "records": provenance_entries,
    }
    prov_path = V2_INDEX_DIR / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Saved caption index → {V2_INDEX_DIR}")
    print(f"Saved provenance    → {prov_path}")
    print(f"\nCaption source breakdown: {caption_source_counts}")
    print(f"Index contains {store.size} record(s) ready for PR2 evaluation.")


def main() -> None:
    from mrta.core.config import settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()

    print(f"=== Build Caption Index ({args.benchmark}) ===\n")

    vlm_model = settings.ollama_vlm_model
    embed_model = settings.embedding_model
    print(f"VLM model  : {vlm_model}")
    print(f"Embed model: {embed_model}")
    print(f"Ollama host: {settings.ollama_host}\n")

    _probe_ollama(settings.ollama_host, vlm_model, embed_model)

    if args.benchmark == "v1":
        _run_v1(vlm_model, embed_model)
    else:
        _run_v2(vlm_model, embed_model)


if __name__ == "__main__":
    main()
