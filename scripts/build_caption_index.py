"""Build a FAISS caption index for the PR1 evaluation corpus.

Extracts figures from the pinned PDF, generates structured visual descriptions
using MRTA's VisualAnalyzer (Qwen2.5-VL), and saves a CaptionVectorStore to
data/eval/indices/caption_index/.

The retrieval text embedded for each figure is the output of
EvidenceRecord.retrieval_text() — caption → detailed_description → nearby_text —
populated by VisualAnalyzer.analyze_evidence(). This is the same text the
production MRTA pipeline would embed; no manually authored captions are used.

Usage:
    python scripts/build_caption_index.py

Requirements:
    - Ollama running with the configured VLM (default: qwen2.5vl:latest)
    - Ollama running with the configured embedding model (default: nomic-embed-text)
    - data/eval/corpus/v1/papers/attention_is_all_you_need.pdf exists

Provenance is recorded in data/eval/indices/caption_index/provenance.json.
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
INDEX_DIR = REPO_ROOT / "data" / "eval" / "indices" / "caption_index"
IMAGES_DIR = INDEX_DIR / "images"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json"


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


def main() -> None:
    from mrta.core.config import settings
    from mrta.core.schemas import EvidenceRecord
    from mrta.ingestion.figure_extractor import extract_figures
    from mrta.multimodal.visual_analyzer import VisualAnalyzer
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.embedder import Embedder

    print("=== Build Caption Index (PR2) ===")
    print()

    if not CORPUS_PDF.exists():
        print(f"ERROR: Corpus PDF not found at {CORPUS_PDF}")
        sys.exit(1)

    vlm_model = settings.ollama_vlm_model
    embed_model = settings.embedding_model
    print(f"VLM model  : {vlm_model}")
    print(f"Embed model: {embed_model}")
    print(f"Ollama host: {settings.ollama_host}")
    print()

    _probe_ollama(settings.ollama_host, vlm_model, embed_model)

    print(f"Extracting figures from {CORPUS_PDF.name} ...")
    figure_records = extract_figures(CORPUS_PDF)
    print(f"  Found {len(figure_records)} figure(s)")
    print()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    analyzer = VisualAnalyzer()
    embedder = Embedder(embed_model)
    store = CaptionVectorStore(embedder)

    evidence_records: list[EvidenceRecord] = []
    provenance_entries: list[dict] = []

    for fig in figure_records:
        print(f"  Analyzing p{fig.page} fig{fig.figure_index} ({fig.width}×{fig.height}) ...")
        ev = fig.to_evidence_record()

        analyzer.analyze_evidence(ev)

        img_path = _save_figure_image(fig, IMAGES_DIR)
        ev.image_path = img_path

        retrieval_text = ev.retrieval_text()
        if not retrieval_text:
            retrieval_text = ev.nearby_text or ""
            ev.nearby_text = retrieval_text

        print(f"    caption  : {ev.caption!r}")
        print(f"    ret_text : {retrieval_text[:80]!r}{'...' if len(retrieval_text) > 80 else ''}")

        evidence_records.append(ev)
        provenance_entries.append(
            {
                "evidence_id": ev.evidence_id,
                "doc_id": ev.doc_id,
                "source": ev.source,
                "page": ev.page,
                "figure_index": ev.figure_index,
                "image_path": img_path,
                "vlm_model": vlm_model,
                "embed_model": embed_model,
                "description_source": "vlm_generated",
                "caption": ev.caption,
                "detailed_description": ev.detailed_description,
                "visual_type": ev.visual_type,
                "retrieval_text_chars": len(retrieval_text),
            }
        )

    print()
    print(f"Embedding and indexing {len(evidence_records)} record(s) ...")
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
    print()
    print(f"Index contains {store.size} record(s) ready for PR2 evaluation.")


if __name__ == "__main__":
    main()
