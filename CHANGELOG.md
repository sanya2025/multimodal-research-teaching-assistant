# Changelog

Records notebook-to-production conversions and significant library changes.
Each entry maps tutorial notebook cells → `src/mrta/` modules → production notebook → unit tests.

---

## [Phase 03] — Embeddings & FAISS — 2026-06-05

**Commit:** pending  
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase03-embeddings-and-faiss.ipynb`  
**Production notebook:** `notebooks/production/2026-05-25-phase03-embeddings-and-faiss.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/retrieval/embedder.py` | Created | New `Embedder` wrapper class — not inline in tutorial; routes to sentence-transformers (`"/" in model_name`) or Ollama REST API |
| `src/mrta/retrieval/vector_store.py` | Created | `VectorStore` extracted from tutorial cell [14] and refined — `__init__(embedder)` drops explicit `dim`; `search()` returns `list[Chunk]` not `list[dict]` |
| `src/mrta/retrieval/__init__.py` | Updated | Exports `Embedder`, `VectorStore` |
| `src/mrta/__init__.py` | Updated | `Embedder`, `VectorStore` added to top-level public API |
| `notebooks/production/…phase03….ipynb` | Updated | Cells [4], [6], [14], [18] switched to `from mrta.*` imports; cells [8], [10], [12], [16] kept inline as teaching demos |
| `tests/unit/test_vector_store.py` | Created | 12 tests — see table below |
| `production-ready.md` | Updated | `retrieval/embedder.py`, `retrieval/vector_store.py` → `✅ done` |
| `notebook-to-production-steps.md` | Updated | Phase 03 section appended |

### Tests created — `tests/unit/test_vector_store.py` (12 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestEmbedder` | `test_embed_returns_correct_shape` | `embed(["hello"])` shape is `(1, dim)` |
| `TestEmbedder` | `test_embed_float32` | output dtype is `float32` |
| `TestEmbedder` | `test_embed_normalised` | L2 norms of all rows are `~1.0` |
| `TestEmbedder` | `test_dim_positive` | `embedder.dim > 0` |
| `TestEmbedder` | `test_model_name_accessible` | `embedder.model_name != ""` |
| `TestVectorStore` | `test_search_returns_k_results` | `search(q, k=1)` returns exactly 1 result |
| `TestVectorStore` | `test_search_returns_chunk_instances` | every result is a `Chunk` instance |
| `TestVectorStore` | `test_search_chunks_have_text` | every result has non-empty `.text` |
| `TestVectorStore` | `test_search_k_capped_at_index_size` | `k > n_chunks` returns `n_chunks`, not `k` |
| `TestVectorStore` | `test_save_load_roundtrip` | reloaded store returns same top chunk for same query |
| `TestVectorStore` | `test_save_creates_expected_files` | `save()` writes `index.faiss`, `metadata.jsonl`, `config.json` |
| `TestVectorStore` | `test_add_empty_is_noop` | `add([])` does not raise; `search()` returns `[]` |

---

## [Phase 02] — Chunking Strategies — 2026-06-05

**Commit:** `836e69a`  
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase02-chunking-strategies.ipynb`  
**Production notebook:** `notebooks/production/2026-05-25-phase02-chunking-strategies.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/core/schemas.py` | Updated | `Chunk` model added — `chunk_id`, `doc_id`, `source`, `page`, `text`, `section?`, `n_tokens?` |
| `src/mrta/ingestion/chunker.py` | Created | `fixed_chunks`, `recursive_chunks`, `token_chunks`, `semantic_chunks`, `chunk_pdf` dispatcher; optional deps imported lazily inside each function body |
| `src/mrta/ingestion/__init__.py` | Updated | Exports `chunk_pdf` alongside `load_pdf` |
| `src/mrta/__init__.py` | Updated | `Chunk`, `chunk_pdf` added to top-level public API |
| `notebooks/production/…phase02….ipynb` | Updated | Cells [4], [6], [8], [10], [12], [17] switched to `from mrta.*` imports; cell [14] (`summarize`) kept inline |
| `tests/unit/test_chunker.py` | Created | 14 tests — see table below |
| `production-ready.md` | Updated | `ingestion/chunker.py` → `✅ done`; `core/schemas.py` updated to note `Chunk` added |
| `notebook-to-production-steps.md` | Updated | Phase 02 section appended |

### Tests created — `tests/unit/test_chunker.py` (14 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestFixedChunks` | `test_returns_chunks` | `len(chunks) > 0` |
| `TestFixedChunks` | `test_all_are_chunk_instances` | every item is a `Chunk` |
| `TestFixedChunks` | `test_chunk_ids_unique` | no duplicate `chunk_id` values |
| `TestFixedChunks` | `test_all_chunks_have_required_fields` | `doc_id`, `source`, `page ≥ 1`, `text.strip()` non-empty |
| `TestFixedChunks` | `test_chunk_id_format` | `_p{page}_c` present in every `chunk_id` |
| `TestFixedChunks` | `test_doc_id_matches_pdf` | chunk `doc_id` == `pdf.doc_id` |
| `TestTokenChunks` | `test_returns_chunks` | `len > 0` (skips if `tiktoken` absent) |
| `TestTokenChunks` | `test_n_tokens_within_size` | every `n_tokens ≤ size` (skips if `tiktoken` absent) |
| `TestTokenChunks` | `test_chunk_ids_unique` | no duplicate IDs (skips if `tiktoken` absent) |
| `TestRecursiveChunks` | `test_returns_chunks` | `len > 0` (skips if `langchain_text_splitters` absent) |
| `TestRecursiveChunks` | `test_chunk_ids_unique` | no duplicate IDs (skips if `langchain_text_splitters` absent) |
| `TestChunkPdf` | `test_default_strategy_is_recursive` | `chunk_pdf(pdf)` returns chunks (default strategy) |
| `TestChunkPdf` | `test_fixed_strategy` | `chunk_pdf(pdf, strategy="fixed")` returns chunks |
| `TestChunkPdf` | `test_unknown_strategy_raises` | `strategy="nonexistent"` raises `ValueError` |

---

## [Phase 01] — PDF Ingestion — 2026-06-05

**Commit:** `013a47c`  
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase01-pdf-ingestion.ipynb`  
**Production notebook:** `notebooks/production/2026-05-25-phase01-pdf-ingestion.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/ingestion/pdf_loader.py` | Updated | `ocr_page_if_needed(page, dpi=200)` extracted as a public function — text fallback or pytesseract OCR for scanned pages |
| `notebooks/production/…phase01….ipynb` | Updated | Cell [18] activated — `from mrta.ingestion.pdf_loader import ocr_page_if_needed` |
| `production-ready.md` | Created | Architecture blueprint — per-phase extraction plan, library map, interface signatures |
| `notebook-to-production-steps.md` | Created | Execution log — Phase 00 and Phase 01 sections; SHA-1 stable ID concept note |

### Tests

`tests/unit/test_ingestion.py` (14 tests) was created in an earlier phase (commit `5db2172`) and
covers `load_pdf` and `_doc_id`. No new test file was added in Phase 01 — `ocr_page_if_needed`
requires a live Ollama or scanned PDF and is tested manually via the production notebook.

---

## Earlier phases — Library scaffold (2026-06-04)

These commits established the package structure before notebook-to-production work began.

| Commit | Description |
|--------|-------------|
| `5db2172` | `tests/unit/test_ingestion.py` — 14 tests for `load_pdf`, `_doc_id` |
| `af53763` | `.github/workflows/ci.yml` — ruff + black + pytest on push/PR |
| `0a5e527` | `docs/adr/` — ADR-001 through ADR-006 |
| `486134e` | `scripts/ingest.py`, `docker/Dockerfile.api`, `docker/docker-compose.yml` |
| `40ac6f5` | `notebooks/production/` — 10 production notebook shells |
| `5046f20` | `src/mrta/__init__.py` public API, VLM model aligned with ADR-003 |
| `34f0070` | `src/mrta/core/config.py` — YAML-backed settings with `MRTA_ENV` switching |
| `3f11477` | `src/mrta/` library layout, `pyproject.toml`, optional dep groups |
