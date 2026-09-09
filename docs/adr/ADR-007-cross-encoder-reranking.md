# ADR-007 — Cross-Encoder Reranking

**Status:** Accepted  
**Date:** 2026-06-11

---

## Context

Dense retrieval (ADR-005) returns the top-k chunks by cosine similarity between the query
embedding and chunk embeddings. Bi-encoder similarity is fast but coarse: it encodes the
query and each chunk independently, so it cannot model fine-grained query-chunk interactions.
In practice this means lower-relevance chunks frequently appear near the top of the retrieved
set, and the LLM either uses them directly (reducing answer precision) or must implicitly
discard them (which it does unreliably).

ADR-005 noted the upgrade path explicitly:

> "Optional reranker: `src/mrta/retrieval/reranker.py` stub — cross-encoder (`bge-reranker-base`)
> is the documented upgrade path."

A cross-encoder processes each (query, chunk) pair jointly, giving a fine-grained relevance
score. Running it over the full corpus is too slow, but running it over a small bi-encoder
candidate set (top-k = 5–20) adds only tens of milliseconds.

## Decision

Implement a `Reranker` class in `src/mrta/retrieval/reranker.py` wrapping
`sentence_transformers.CrossEncoder`. Wire it into `rag_query()` as an optional parameter
so callers can enable reranking without breaking existing code.

### Model

Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Rationale:
- Trained on MS MARCO passage ranking — directly applicable to research-paper QA.
- 6 transformer layers, ~22M parameters — inference over 5–10 pairs takes ~30–50 ms on CPU.
- Available via `sentence-transformers`, which is already a core dependency — no new package.
- Matches the `bge-reranker-base` upgrade path described in ADR-005 in spirit; MiniLM-L-6-v2
  was chosen over bge-reranker because it needs no extra tokenizer configuration and is well
  tested in the sentence-transformers ecosystem.

### Interface

```python
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None: ...
    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 3) -> list[Chunk]: ...
```

`Reranker` is optional in `rag_query()`:

```python
def rag_query(
    question: str,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = 5,
    reranker: Reranker | None = None,
    rerank_top_n: int = 3,
) -> dict: ...
```

When `reranker=None` (default), behaviour is identical to before this change. When provided,
bi-encoder top-k results are re-scored and the top-n highest-ranked chunks are passed to the
LLM instead.

### Testing

`CrossEncoder` is mocked in all unit tests via `patch("sentence_transformers.CrossEncoder")`.
The real model is never downloaded in CI.

## Consequences

**Positive:**

- Answer precision improves on queries where the most semantically relevant chunk is not the
  closest by embedding cosine similarity (common for paraphrased or abstractive questions).
- No new dependency — `sentence-transformers` is already in `dependencies`.
- Fully optional — zero impact on existing callers, the API endpoint, or the Streamlit app.
- Model is swappable via `model_name` parameter; upgrading to a larger cross-encoder requires
  no code change.

**Negative / Tradeoffs:**

- Adds latency proportional to `top_k`: ~30–50 ms per query on CPU for the default model.
  This is acceptable for interactive use but should be profiled before enabling in production.
- `rerank_top_n` (default 3) is fewer chunks than `top_k` (default 5), so the LLM sees less
  context per query. If the cross-encoder mis-ranks a key chunk outside top-n, recall drops.
  Tuning `top_k` and `rerank_top_n` together is a future task.
- Model download on first use (~85 MB) — not a concern for local dev but matters for cold-start
  containers. The Docker image should pre-download the model in a future image-build step.

## PR5 extension — reranking multimodal fused candidates (evaluation only)

**Date:** 2026-09-08 · **Status:** Accepted, additive

PR5 extends this decision from text-only RAG to the multimodal retrieval evaluation
path. The model, library and mocking convention above are unchanged; nothing in the
production `Reranker` or `rag_query()` is modified, and PR5 adds no production
pipeline integration.

### What is added

`CrossEncoderReranker` in the same module, consuming the `FusedCandidate` lists
produced by `reciprocal_rank_fusion_canonical()` (PR4) and returning
`RerankedCandidate`:

```text
Text ──────┐
Caption ───┼→ canonical RRF → top-20 → CrossEncoder → top-5
CLIP ──────┘
```

Two rerankers now live in `reranker.py` for the same reason two fusion APIs live in
`fusion.py`: they consume different types and answer different questions. The
production `Reranker` returns bare `Chunk` objects; PR5's diagnostics need the RRF
score and RRF rank preserved alongside the cross-encoder score, so
`FusedCandidate.score` is never overwritten and the candidate is never mutated.

RRF and cross-encoder scores are **not blended**. RRF orders the candidate set
entering the reranker; the cross-encoder is the final ranking stage.

### Candidate text representation

The model is a text cross-encoder and cannot inspect image pixels, so figure
candidates must be represented textually. `VisualRecord` (the CLIP index record)
carries no text at all, only an `image_path`. Figure text therefore comes from a
single `canonical_id → text` lookup built once from the frozen PR2 caption index and
shared by the caption and CLIP streams, so a figure's representation does not depend
on which stream retrieved it:

```text
text     ->  the retrieved chunk's own text (never page-level text)
figure   ->  "Figure caption: {VLM caption or extracted caption}"
             "Description: {VLM detailed_description}"
             "Context: {nearby-text fallback}"
```

A canonical figure owning several caption records (multi-crop) resolves
deterministically: prefer a record with a VLM caption, then the lowest `evidence_id`.

Candidate text reaches the model only through `FusedCandidate.payload`, and only
production-derived keys are ever written there, so benchmark ground truth is
structurally unable to enter model input.

### Consequences measured on the frozen v2 benchmark

**Positive:** overall MRR@5 rose 0.2658 → 0.5153 and Recall@5 0.3400 → 0.6050 against
three-stream RRF. CLIP contributed complementary visual candidates that equal-weight RRF
could not rank effectively on the frozen v2 benchmark, costing −0.1037 MRR under fusion
alone. Query-aware reranking converts part of that complementary signal into measurable
gains: +0.0450 MRR, +0.0800 Recall@5 and +0.1067 Figure Recall@5. The finding is
conditional on reranking following fusion; it is not evidence that equal-weight RRF is
adequate for CLIP on its own.

**Negative:** figure preservation remains unresolved. Overall Figure Recall@5 decreases
from 0.5467 to 0.4800, but the aggregate hides a large representation-quality
interaction: figures with VLM captions improve to 0.7027, whereas nearby-text fallback
figures fall to 0.2632. The reranker cannot compensate for a figure whose only textual
representation is page-extraction boilerplate; on the available evidence this is a
representation limit rather than a ranking one, and it argues for improving figure
captioning before considering a multimodal reranker.

**Latency** (local, warm model, Apple Silicon MPS): 43.9 ms mean, 37.6 ms p50, 70.5 ms
p95 per query for 20 candidate pairs (~456 pairs/s), excluding an 11.7 s model load
and 0.44 s first-inference warm-up. Consistent with the 30–50 ms estimate above. Not a
production SLO claim.

### Related

- [ADR-008 — Multimodal RAG Architecture](ADR-008-multimodal-rag-architecture.md)
- `results/v2/pr5_reranker_metrics.json` — measured results and limitations

---

## References

- [MS MARCO Passage Ranking benchmark](https://microsoft.github.io/msmarco/)
- [sentence-transformers cross-encoder docs](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [Nogueira & Cho, 2019 — Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- [ADR-005 — RAG Architecture](ADR-005-rag-architecture.md)
- [ADR-004 — Embedding Model Selection](ADR-004-embedding-model-selection.md)
