# ADR-005 — RAG Architecture

**Status:** Accepted  
**Date:** 2026-06-04

---

## Context

Retrieval-Augmented Generation (RAG) for research PDFs requires decisions across four layers: ingestion, chunking, retrieval, and generation. Each choice affects answer quality, latency, and tutorial teachability.

## Decision

### Ingestion

Use **PyMuPDF (`fitz`)** as the primary PDF parser:

- Page-accurate text extraction with layout metadata.
- Direct image extraction per page (needed for multimodal path).
- Fast enough for real-time use in tutorials.

`pdfplumber` is available as a fallback for complex table layouts.

Schema: every page produces a `PageRecord(doc_id, page, text, n_images, source)`. Documents are `PdfDocument(doc_id, source, title, n_pages, pages)`. Defined in `src/mrta/core/schemas.py`.

### Chunking

**Page-aware recursive character splitting** (LangChain `RecursiveCharacterTextSplitter`):

- Default: `chunk_size=1000`, `chunk_overlap=200` (dev); `chunk_size=500`, `chunk_overlap=50` (test).
- Metadata preserved per chunk: `{doc_id, page, section, chunk_id}`.
- Page boundary respected: chunks do not span pages, preserving citation accuracy.

Rationale for 1000 tokens (dev): research papers have dense paragraphs; smaller chunks lose cross-sentence context. 700–1000 tokens is the empirical sweet spot for academic text (revisited in Notebook 02).

### Retrieval

**Dense retrieval only** (Phase 1). Hybrid retrieval (BM25 + dense) is documented as a future improvement.

Top-k: `settings.top_k = 5` (dev). Each retrieved chunk carries full metadata for citation.

Optional reranker: `src/mrta/retrieval/reranker.py` stub — cross-encoder (`bge-reranker-base`) is the documented upgrade path.

### Generation

Prompt includes retrieved chunks with page metadata. LLM is instructed to cite specific pages in every claim.

Citation verification in eval pipeline: cited page numbers are checked against retrieved chunks to catch hallucinated citations.

### Multimodal Extension

Figures extracted per page as PNG bytes. CLIP embeds figures into the same vector space as text. At query time, both text and image indexes are searched; results are merged and re-ranked by score.

VLM (Qwen2.5-VL or LLaVA) captions figures on demand. Captions are returned alongside text answers when figures are retrieved.

## Consequences

**Positive:**

- Citation metadata at every step enables verifiable, grounded answers.
- Modular design: each layer can be swapped independently.
- Multimodal path is additive — text-only RAG works without a VLM.

**Negative / Tradeoffs:**

- No hybrid retrieval (BM25) in Phase 1; keyword-heavy queries may underperform.
- Page-boundary chunking can split arguments that span pages.
- CLIP image-text space alignment is weaker than purpose-built multimodal retrievers.

## References

- [RAG paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Lost in the Middle — retrieval position effects](https://arxiv.org/abs/2307.03172)
- [Notebook 04 — RAG Pipeline](../../notebooks/tutorials/2026-05-25-phase04-rag-pipeline.ipynb)
