# Notebook-to-Production Steps

## Purpose

This is the **execution log** for converting tutorial notebooks into production library code.
It records — notebook by notebook — which cells run, which functions were extracted, and what
the production notebook looks like after each phase.

Use it to answer: *"What has already been done, and where does the next session pick up?"*

It does **not** define what the library should look like when complete — that lives in
[production-ready.md](production-ready.md), the architecture blueprint.

---

Per-notebook guide: what to run, what to extract, and what the production version looks like.

---

## Phase 00 — Foundations & Setup

Notebook 00 is purely a setup verification notebook — no code to write. Here's what each section does when you run it:

| Cell | What it does | Requires |
|------|--------------|----------|
| [0]  | Stale TODO — already done (package exists) | Delete or ignore |
| [4]  | Prints Python version and platform | Nothing |
| [6]  | Checks Ollama is running at localhost:11434 | `ollama serve` running |
| [8]  | First LLM call — `llama3.2:latest` | `ollama pull llama3.2:3b` |
| [11] | Downloads MiniLM embedder from HF | Internet (first run only) |
| [12] | Embedding call — `nomic-embed-text` | `ollama pull nomic-embed-text` |
| [14] | Checks all key imports work | `pip install -e ".[all]"` |
| [16] | Loads `mrta.settings` and prints config values | Nothing (already installed) |

**So: run it top to bottom.** If all cells pass without error, your environment is ready for Phase 01 onward.

The one thing to clean up first: delete cell [0] (the TODO). It was the pre-Phase-1 reminder to build the package — that's done. The production notebook already has it removed; the tutorial notebook still has it as a stale artifact.

---

## Phase 01 — PDF Ingestion

Phase 01 is the first notebook with real library code. Two functions have already been extracted to
`src/mrta/` and are fully working — the production notebook imports them directly.

### What's extracted (production imports active)

| Tutorial cell | Inline code | Production import |
|---------------|-------------|-------------------|
| [6]  | `class PageRecord(BaseModel): ...` `class PdfDocument(BaseModel): ...` | `from mrta.core.schemas import PageRecord, PdfDocument` |
| [8]  | `def _doc_id(path): ...` `def load_pdf(path): ...` | `from mrta.ingestion import load_pdf` |
| [17] | `def ocr_page_if_needed(page, dpi): ...` | `from mrta.ingestion.pdf_loader import ocr_page_if_needed` |

### What's still inline (extracted in a later phase)

| Tutorial cell | Inline code | Extracted when |
|---------------|-------------|----------------|
| [12–14] | Figure saving loop + PIL display | Phase 07 — becomes `src/mrta/ingestion/figure_extractor.py` |

### Running the production notebook

The production notebook (`notebooks/production/`) is **fully runnable as-is**.

| Cell | What it does | Status |
|------|--------------|--------|
| [3]  | Loads PDF from `data/sample/` — no download | ✅ runs |
| [5]  | Opens PDF with PyMuPDF, prints metadata | ✅ runs |
| [7]  | Imports `PageRecord`, `PdfDocument` from `mrta.core.schemas` | ✅ runs |
| [9]  | Imports and calls `load_pdf()` from `mrta.ingestion` | ✅ runs |
| [11] | Builds a pandas DataFrame of pages | ✅ runs |
| [13] | Extracts and saves embedded figures to `data/processed/figures/` | ✅ runs (inline) |
| [15] | Displays one figure inline | ✅ runs |
| [18] | Imports `ocr_page_if_needed` from `mrta.ingestion.pdf_loader` | ✅ runs |

**Requires:** `pip install -e ".[all]"` and `data/sample/attention_is_all_you_need.pdf` (already git-tracked).

### Functions implemented in Phase 01

All three live in `src/mrta/ingestion/pdf_loader.py`. The schemas live in `src/mrta/core/schemas.py`.

| Function / class | Signature | What it does |
|-----------------|-----------|--------------|
| `_doc_id` | `(path: Path) -> str` | SHA-1 hash of file bytes → stable `"{stem}_{hash[:10]}"` ID |
| `load_pdf` | `(path: str \| Path) -> PdfDocument` | Opens PDF, extracts text + image count per page, returns typed `PdfDocument` |
| `ocr_page_if_needed` | `(page: fitz.Page, dpi: int = 200) -> str` | Returns embedded text if present; falls back to pytesseract OCR for scanned pages |
| `PageRecord` | Pydantic model | One page: `doc_id`, `page`, `text`, `n_images`, `source` |
| `PdfDocument` | Pydantic model | Full document: `doc_id`, `source`, `title`, `n_pages`, `pages: list[PageRecord]` |

### Concept: SHA-1 as a stable document ID

SHA-1 is a hashing algorithm — feed it any bytes, get a fixed 40-character hex string back.
The same input always produces the same output; any change to the content produces a
completely different output.

`_doc_id` hashes the raw PDF bytes so the ID is tied to the file content, not the filename:

```python
def _doc_id(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}_{h}"
```

| File | Content changed? | `doc_id` |
|------|-----------------|----------|
| `attention_is_all_you_need.pdf` | — | `attention_is_all_you_need_a3f2c91b` |
| `attention_is_all_you_need.pdf` | no (same file) | `attention_is_all_you_need_a3f2c91b` (identical) |
| `paper_renamed.pdf` | no (just renamed) | `paper_renamed_a3f2c91b` (same hash suffix) |
| `attention_is_all_you_need_v2.pdf` | yes | `attention_is_all_you_need_v2_7d4e110f` (different) |

**Why this matters for RAG:** every `PageRecord` stores `doc_id`. When the LLM cites page 7,
the system resolves `doc_id` back to the source file. If the ID changed between runs,
all citations would break. Hashing the content guarantees stability as long as the file
doesn't change.

---

## Phase 02 — Chunking Strategies

Phase 02 introduces the `Chunk` schema and four chunking strategies. All five functions
plus the dispatcher have been extracted to `src/mrta/ingestion/chunker.py`.

### What's extracted — Phase 02 (production imports active)

| Tutorial cell | Inline code | Production import |
|---------------|-------------|-------------------|
| [4] | `class PageRecord/PdfDocument`, `load_pdf` (re-defined) | `from mrta.core.schemas import PageRecord, PdfDocument` + `from mrta.ingestion import load_pdf` |
| [6] | `class Chunk(BaseModel): ...` | `from mrta.core.schemas import Chunk` |
| [8] | `def fixed_chunks(pdf, size, overlap): ...` | `from mrta.ingestion.chunker import fixed_chunks` |
| [10] | `def recursive_chunks(pdf, size, overlap): ...` | `from mrta.ingestion.chunker import recursive_chunks` |
| [12] | `def token_chunks(pdf, size, overlap): ...` | `from mrta.ingestion.chunker import token_chunks` |
| [15] | `def semantic_chunks(pdf, threshold, max_chars): ...` | `from mrta.ingestion.chunker import semantic_chunks` |

### What's still inline (stays in notebook)

| Tutorial cell | Inline code | Why it stays |
|---------------|-------------|--------------|
| [14] | `def summarize(name, chunks): ...` | Notebook display helper — not reusable library code |

### Running the Phase 02 production notebook

| Cell | What it does | Status |
|------|--------------|--------|
| [4] | Imports `PageRecord`, `PdfDocument`, `load_pdf`; loads from `data/sample/` | ✅ runs |
| [6] | Imports `Chunk` from `mrta.core.schemas` | ✅ runs |
| [8] | Imports + calls `fixed_chunks()` | ✅ runs |
| [10] | Imports + calls `recursive_chunks()` | ✅ runs |
| [12] | Imports + calls `token_chunks()` | ✅ runs |
| [14] | Inline `summarize()` + comparison DataFrame | ✅ runs (inline) |
| [17] | Imports + calls `semantic_chunks()` on first 5 pages | ✅ runs |
| [19] | Saves `tokenized` chunks to `data/processed/chunks.jsonl` | ✅ runs |

**Requires:** `pip install -e ".[all]"` and `data/sample/attention_is_all_you_need.pdf`.

### Functions implemented in Phase 02

Functions live in `src/mrta/ingestion/chunker.py`. `Chunk` lives in `src/mrta/core/schemas.py`.

| Function / class | Signature | What it does |
|-----------------|-----------|--------------|
| `Chunk` | Pydantic model | One chunk: `chunk_id`, `doc_id`, `source`, `page`, `text`, `section?`, `n_tokens?` |
| `fixed_chunks` | `(pdf, size=1500, overlap=200) -> list[Chunk]` | Naive fixed-width character windows |
| `recursive_chunks` | `(pdf, size=800, overlap=100) -> list[Chunk]` | LangChain splitter — respects paragraph/sentence boundaries |
| `token_chunks` | `(pdf, size=700, overlap=100) -> list[Chunk]` | tiktoken-based — guarantees chunks fit a token budget |
| `semantic_chunks` | `(pdf, threshold=0.55, max_chars=1200) -> list[Chunk]` | Merges sentences by embedding similarity — slow, highest quality |
| `chunk_pdf` | `(pdf, strategy="recursive", **kwargs) -> list[Chunk]` | Dispatcher — routes to the right strategy by name |

### Concept: chunk_id encodes provenance

Every chunk carries a `chunk_id` of the form `{doc_id}_p{page}_c{idx}`:

```text
attention_is_all_you_need_a3f2c91b_p3_c2
│                            │      │  │
└─ stem                      └─hash │  └─ chunk index on this page
                                    └─ page number (1-indexed)
```

This means the RAG system can resolve any citation back to the exact page and chunk without
keeping a separate lookup table — the location is encoded in the ID itself.
