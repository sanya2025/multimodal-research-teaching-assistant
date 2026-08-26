# ADR-008 — Multimodal RAG Architecture

**Date:** 2026-08-26
**Status:** Accepted
**Branch:** `feature/mmrag-evaluation` (Stage 7) — completes Stages 1–8

---

## Context

The text-only RAG pipeline (`rag_query`, `VectorStore`, `LLMClient`) handles
text documents well. A multimodal research assistant also needs to retrieve
and reason over embedded figures and page images.

This ADR documents the complete multimodal RAG architecture added across
Stages 1–8 of the multimodal implementation.

---

## Decisions

### 1. Separate embedding spaces, fusion at ranking level

Three independent stores operate in different embedding spaces:

| Store | Embedding | Dimensionality |
|---|---|---|
| `VectorStore` | sentence-transformer | 384 |
| `CaptionVectorStore` | sentence-transformer (on VLM captions) | 384 |
| `VisualVectorStore` | CLIP image embeddings | 512 |

Raw cosine scores from different spaces are never compared directly.
Only rank order feeds Reciprocal Rank Fusion (RRF, k=60).

**Rationale:** Cross-space score comparison is theoretically unsound. RRF is
parameter-light, deduplicates by `evidence_id`, and degrades gracefully when
one or two stores are empty or absent.

### 2. `EvidenceRecord` as the common currency

All retrieval paths produce `EvidenceRecord` objects. This gives a unified
schema that carries modality, provenance, image bytes, and textual
representations across retrieval, generation, and evaluation.

### 3. `MultimodalRAG` as the single generation entry point

`MultimodalRAG.ask(question, teaching_mode)` wires retrieval → fusion →
prompt rendering → VLM call → structured citations. Teaching modes
(`explain`, `socratic`, `quiz`, `compare`, `visual_evidence`) select
different Jinja2 templates; the retrieval, citation, and fallback logic
are identical across modes.

### 4. VLM fallback to text-only on `LLMError`

When `VLMClient.generate()` raises `LLMError` (model unavailable or does
not support vision), the system retries with the same text evidence but no
images attached and sets `retrieval_mode="text_only"` in the returned
`MultimodalAnswer`. Text RAG continues to work in all cases.

### 5. No binary image bytes in API responses

`VisualSource` in the API response carries `(label, page, source, figure_index,
modality)` only. Streamlit fetches thumbnails via a separate `/figures` call.
This keeps response sizes predictable and avoids base64 bloat in JSON.

### 6. Evaluation metrics report modalities separately

`multimodal_recall_at_k` returns text, visual, and overall recall as separate
numbers. A single aggregate would mask failures in one modality — e.g. perfect
text recall + zero figure recall would show as 0.5 overall with a mean but 0
with the geometric mean we use.

### 7. Optional multimodal dependencies

`mrta-rag[multimodal]` (open_clip, Pillow, sentence-transformers cross-encoder)
is optional. The API lifespan tries to initialise the multimodal stack and
silently falls back when the extra is absent. Text RAG remains fully functional.

---

## Consequences

- Adding a new retrieval modality requires a new `*VectorStore` and a new
  `named_lists` key in `MultimodalRetriever` — no other files change.
- Teaching modes are additive: add a Jinja2 template and register it in
  `VALID_TEACHING_MODES`. No changes to the retrieval or citation pipeline.
- Evaluation is decoupled from retrieval: `multimodal_metrics.py` operates
  on `EvidenceRecord` lists and `MultimodalAnswer` objects — it does not
  touch stores or models.

---

## Alternatives Considered

| Alternative | Rejected because |
|---|---|
| Single unified embedding space | CLIP and sentence-transformer spaces are not compatible; joint embedding requires fine-tuning |
| Late fusion (average scores) | Dimensionality mismatch; ranking fusion is theoretically cleaner |
| Separate `TeachingRAG` class | Redundant — same retrieval, different template; one class with a parameter is simpler |
| Binary image bytes in API JSON | Unpredictable response sizes; breaks streaming; forces clients to handle large payloads |

---

## Related ADRs

- ADR-002 — Vector store: FAISS chosen for text store (same applies to visual store)
- ADR-005 — RAG architecture: text-only pipeline unchanged by multimodal extension
- ADR-006 — Evaluation framework: extended here to cover multimodal metrics
