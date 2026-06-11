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

## References

- [MS MARCO Passage Ranking benchmark](https://microsoft.github.io/msmarco/)
- [sentence-transformers cross-encoder docs](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [Nogueira & Cho, 2019 — Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- [ADR-005 — RAG Architecture](ADR-005-rag-architecture.md)
- [ADR-004 — Embedding Model Selection](ADR-004-embedding-model-selection.md)
