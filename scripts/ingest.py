"""CLI script to ingest a PDF and print a summary.

Usage:
    python scripts/ingest.py <path-to-pdf>

Example:
    python scripts/ingest.py data/raw/attention_is_all_you_need.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from mrta.ingestion.pdf_loader import load_pdf


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest.py <path-to-pdf>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)
    if path.suffix.lower() != ".pdf":
        print(f"Error: expected a .pdf file, got: {path.suffix}")
        sys.exit(1)

    print(f"Loading {path.name} ...")
    doc = load_pdf(path)

    total_chars = sum(len(p.text) for p in doc.pages)
    pages_with_images = [p.page for p in doc.pages if p.n_images > 0]

    print()
    print(f"  doc_id   : {doc.doc_id}")
    print(f"  title    : {doc.title or '(none)'}")
    print(f"  source   : {doc.source}")
    print(f"  pages    : {doc.n_pages}")
    print(f"  chars    : {total_chars:,}")
    print(f"  pages with images : {pages_with_images or 'none'}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
