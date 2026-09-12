# ADR-009 — Canonical Retrieval in the Production Query Path

**Date:** 2026-09-09
**Status:** Accepted
**Branch:** `feat/pipeline-multimodal-generation` (PR6)
**Amends:** [ADR-008](ADR-008-multimodal-rag-architecture.md) decisions 3 and 5

---

## Context

PR1–PR5 built and measured a multimodal retrieval stack against the frozen v2
benchmark, ending at:

```text
Text ──────┐
Caption ───┼──► canonical RRF ──► top-20 ──► CrossEncoder ──► top-5
CLIP ──────┘
```

On v2 that stack measured Recall@5 0.6050 / MRR@5 0.5153, against 0.3400 /
0.2658 for the same three-stream fusion without reranking
(`results/v2/pr5_reranker_metrics.json`).

None of it was reachable from production. The production multimodal path
(`MultimodalRetriever` → `MultimodalRAG`, ADR-008) used the *legacy* fusion API:

- `reciprocal_rank_fusion` keyed on `evidence_id`, not PR4's canonical identity;
- reranking applied to an already-truncated 8-item fused list, via the private
  `Reranker._model`, discarding score provenance;
- the API lifespan never passed a `caption_store`, so the caption stream — the
  strongest single visual signal in PR2 — was dead in production;
- `VisualVectorStore` rather than the `ImageStore` PR3 evaluated.

So the measured system and the shipped system were different systems.

### The blocker: canonical identity without a manifest

PR4/PR5 key figure evidence on `figure_id`, which `EvalAdapter` resolves from
the frozen v2 benchmark manifest. Production has no manifest, and production
code must not depend on evaluation artifacts.

It does not need one. Ingestion already assigns every figure the stable triple
`(doc_id, page, figure_index)` — exactly what `FigureRecord.to_evidence_record()`
encodes into `evidence_id`.

## Decisions

### 1. A production canonical adapter, deriving figure identity from ingestion

`mrta.retrieval.canonical_pipeline` derives `figure_id` as `p{page}_f{index}`
from data ingestion already produces. PR4's `canonical_identity()` then applies
unchanged, so a caption hit and a CLIP hit on one physical figure collapse into
a single candidate accumulating both streams' RRF contributions.

The benchmark's *semantic* figure ids (`fig_transformer_arch`) are deliberately
not used: they exist only in the eval manifest. A `VisualRecord`'s persisted
`figure_id` is ignored in favour of the derived one, so identity is consistent
regardless of which store produced the record.

### 2. `retrieve_multimodal()` as the single retrieval orchestration point

One typed function returning `list[RerankedCandidate]` plus diagnostics. It
composes existing components and reimplements none: fusion is
`reciprocal_rank_fusion_canonical`, reranking is `CrossEncoderReranker`. RRF
score, RRF rank, cross-encoder score and reranker rank remain separate
throughout; `FusedCandidate.score` is never overwritten.

Parameters stay at the evaluated values — pool 20, `rrf_k` 60, final top-5 —
and are not tuned in production.

### 3. `CanonicalMultimodalRAG` alongside `MultimodalRAG` (amends ADR-008 §3)

ADR-008 named `MultimodalRAG` the *single* generation entry point. That no
longer holds: there are now two, consuming different evidence types.
`MultimodalRAG` takes `EvidenceRecord` lists from the legacy retriever;
`CanonicalMultimodalRAG` takes `RerankedCandidate` carrying canonical identity
and ranking provenance.

They are separate rather than merged because the types differ at every step —
identity, dedup semantics, and score provenance — and because teaching-mode
templates consume `EvidenceRecord` and would have to be rewritten to merge
them. This mirrors the precedent set in `fusion.py` (PR4) and `reranker.py`
(PR5), where the evaluated implementation was added beside the legacy one.

`/ask` prefers the canonical path when a canonical stack is configured and no
`teaching_mode` is requested; teaching modes stay on the legacy path so their
rendered prompts are unchanged.

### 4. Application-assigned citation labels, verified against retrieved evidence

The application assigns `[T1]`/`[F1]` labels and resolves model output back to
them. Labels the model invents resolve to nothing and are reported rather than
parsed into citations. Citations are constructed from retrieved evidence only,
so a fabricated page, figure id or file path cannot reach a response.

This replaces trusting model-emitted citation metadata.

### 5. Image paths are validated metadata, never model input (amends ADR-008 §5)

ADR-008 §5 fixed `VisualSource` at `(label, page, source, figure_index,
modality)`. PR6 adds `document_id`, `figure_id`, `image_path`, `caption` and
`modality_sources`. The no-binary-bytes principle is unchanged — `image_path`
is a relative reference, not base64.

An emitted `image_path` must be relative, stay inside the `data/` asset tree,
and exist on disk; otherwise it is dropped and the figure keeps its textual
evidence. A path is never placed in the prompt: a filesystem path carries no
visual information, and presenting one as if it did would invite the model to
describe an image it cannot see.

### 6. Per-stream graceful degradation

Text retrieval is required — it is the only index production ingestion always
builds. Caption, CLIP and the reranker each degrade independently: an absent or
failing component removes its contribution, records the reason in the
diagnostics and trace span, and the query proceeds. A reranker failure falls
back to canonical RRF ordering, marked `reranker_text_source="not_reranked"` so
an unreranked result is never mistaken for a reranked one.

### 7. Production ingestion builds all three indices

`mrta.ingestion.document_indexer.index_document()` is the single ingestion entry
point. One uploaded PDF yields the text index, the caption index and the CLIP
index, plus figure PNGs under `data/figures/`. All three persist to the same
`vector_store_path` root and are reloaded by the API lifespan.

The caption record and the CLIP record for one figure are built from the same
`FigureRecord`, so they share a canonical identity by construction rather than by
coincidence. Text indexing happens first and unconditionally: visual work can
never cost a document its text retrieval.

### 8. Canonical retrieval gated by configuration

`enable_canonical_retrieval` and `enable_cross_encoder_rerank` default to true
but are false under `MRTA_ENV=test`, so the suite and CI never load or download
cross-encoder weights. Loading them eagerly in the API lifespan took the API
test module from 0.3 s to 104 s.

## Consequences

**Positive**

- The measured stack and the shipped stack are the same stack.
- Caption retrieval reaches production for the first time.
- Text and figure evidence stay semantically distinct end to end, and one
  physical figure found by two streams yields one citation.
- Citation provenance (`modality_sources`, `rrf_rank`, `reranker_rank`) is
  available for debugging without exposing raw scores.

**Negative / tradeoffs**

- Two multimodal generation paths now exist. Justified while teaching modes
  depend on the legacy templates, but it is duplication and should collapse
  once teaching modes are ported.
- Ingestion now captions every extracted figure with the VLM, so upload latency
  scales with figure count (~45 s for a 15-page paper with 3 figures locally).
  Uploads are synchronous; a background job would suit a larger corpus.
- The derived `figure_id` is not stable across a re-ingestion that changes
  figure extraction order, since it is positional.
- Cross-encoder loading adds ~12 s to API startup when enabled.

## Alternatives considered

**Modify `MultimodalRetriever` in place.** Rejected: it would change the
behaviour of every existing caller and test at once, and PR6 has a hard
backward-compatibility requirement.

**Reuse `EvalAdapter` in production.** Rejected: it requires the benchmark
manifest, coupling production to evaluation artifacts.

**Have the model emit page/figure metadata and parse it.** Rejected: it makes
hallucinated provenance indistinguishable from real provenance.

## Related ADRs

- [ADR-005 — RAG Architecture](ADR-005-rag-architecture.md)
- [ADR-006 — Evaluation Framework](ADR-006-evaluation-framework.md)
- [ADR-007 — Cross-Encoder Reranking](ADR-007-cross-encoder-reranking.md)
- [ADR-008 — Multimodal RAG Architecture](ADR-008-multimodal-rag-architecture.md)
