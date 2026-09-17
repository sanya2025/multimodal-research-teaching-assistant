# ADR-013 — Preserve Modality-Specific Retrieval and Query-Aware Reranking Based on v2 Evidence

**Date:** 2026-09-16
**Status:** Accepted
**Branch:** `docs/portfolio-research-overhaul` (PR10)
**Relates to:**
[ADR-002](ADR-002-vector-store-faiss-vs-qdrant.md),
[ADR-005](ADR-005-rag-architecture.md),
[ADR-007](ADR-007-cross-encoder-reranking.md),
[ADR-008](ADR-008-multimodal-rag-architecture.md),
[ADR-009](ADR-009-canonical-retrieval-production-integration.md),
[ADR-010](ADR-010-ablation-and-generation-evaluation.md),
[ADR-011](ADR-011-citation-aware-generation.md),
[ADR-012](ADR-012-two-tier-ci-quality-gates.md)

---

## Context

The architectural question running through PR1–PR9 was how to combine three
retrieval signals over research PDFs:

- semantic **text** retrieval over body chunks;
- **figure-caption** retrieval over VLM-generated and fallback figure text;
- direct **CLIP** visual retrieval over figure images.

This ADR does not introduce a new decision. It consolidates the architecture the
frozen v2 experiments produced, and records the evidence for each choice so the
architecture can be audited rather than taken on trust.

The evidence is specific to frozen v2 — 100 queries, 5 papers, 21 canonical
figures — which informed development and is not an untouched held-out test set.
Full detail: [`../architecture/multimodal_reranking_v2.md`](../architecture/multimodal_reranking_v2.md).

The central finding is that the same architectural change had **opposite signs**
depending on what ranked its output:

| | Text+Caption | + CLIP | ΔMRR@5 |
|---|---:|---:|---:|
| Equal-weight RRF | 0.3695 | 0.2658 | **−0.1037** |
| After cross-encoder reranking | 0.4703 | 0.5153 | **+0.0450** |

## Decisions

### 1. Preserve modality-specific retrieval streams

Each modality keeps its own index, encoder, and candidate list. No stream is
folded into another.

Evidence: each single-stream configuration wins on a different axis — caption-only
leads Figure Recall@5 (0.5467), text-only leads single-stream MRR@5 (0.3292), and
CLIP-only reaches Figure Recall@5 0.4800 with the weakest MRR@5 (0.1983). They are
complementary, not redundant.

### 2. Never blend raw similarity scores across modalities

Text and caption embeddings occupy a 768-d `nomic-embed-text` space; CLIP occupies
a 512-d space. Cosines from these spaces are not commensurable — equal numeric
values do not represent equal relevance.

Fusion therefore operates on **ranks**, not scores. This is the direct reason RRF
was chosen over score normalization: rank fusion needs no assumption about the
shape or scale of either distribution.

### 3. Apply canonical evidence identity and deduplication

Identity is `(document_id, page_number, figure_id)`. A figure reachable through
both the caption and CLIP streams is **one** piece of evidence.

Without this, a figure found twice would occupy two of five final slots and be
counted twice at every cutoff — metrics would improve while the system got worse.
Deduplication is a correctness requirement of the measurement, not an
optimization.

### 4. Retain equal-weight RRF as a transparent first-stage fusion mechanism

Equal weights, `k_rrf = 60`, per-stream depth 20, deterministic tie-breaking.

No weight was tuned against v2. A tuned weight would have hidden the PR4 negative
result — the most informative measurement in the project — behind a parameter,
and would have converted an ablation into a search. RRF's job is to assemble a
*reachable candidate pool*, not to produce the final order.

### 5. Apply query-aware cross-encoder reranking to the fused pool

`cross-encoder/ms-marco-MiniLM-L-6-v2` over the fused top-20, emitting top-5. The
CE score is stored separately and **never blended** with the RRF score.

Evidence: the reranker moved **all 38** text targets into the top-5, from mean
rank 15.89 to 1.58 (median delta −14), lifting `text`-intent MRR@5 from 0.0000 to
0.6713. The cross-encoder is modality-blind and query-aware, so it scores a
displaced text chunk on its merits rather than on how many streams found it.

### 6. Preserve CLIP despite its negative marginal effect under three-stream RRF

The tempting conclusion after PR4 was to drop CLIP. The evidence says the
opposite once ranking changed.

CLIP's contribution is **candidate generation**, not ranking. It cut
`retrieval_miss` from 24 to 7 of 100 queries, and raised the count of PR4's
in-pool failures that were *reachable within depth 20* from 10 to 17. A stream
that makes targets reachable is valuable even when the fusion stage cannot order
them — provided a later stage can.

**Narrow reading.** On frozen v2, CLIP supplied complementary visual candidates
that equal-weight RRF could not rank effectively; query-aware reranking converted
part of that signal into measurable gains. This is **not** evidence that RRF
generally fails, that CLIP is universally beneficial, or that CLIP only works with
reranking.

### 7. Treat figure extraction and textual representation as an independent bottleneck

A large measured performance gap is associated with figure-text provenance: in
the full reranked configuration, **Figure Recall@5 was 0.7027 for VLM-caption
targets (n=37 queries) versus 0.2632 for nearby-text-fallback targets (n=38
queries)**. This gap is larger than any difference the frozen experiments
produced by changing fusion or reranking.

Many scientific figures are vector graphics that raster extraction cannot
recover, so they enter the caption index through adjacent page prose. The
grouping is observational — figures are classified by how their text was
obtained, not randomly assigned — so the experiments establish an association,
not a cause. Treating extraction and textual representation as an independent
workstream follows from the size of that association; whether a different
reranker would narrow it was not evaluated.

### 8. Evaluate retrieval separately from generation and citation behaviour

Higher retrieval metrics did not necessarily translate into higher citation F1
across the evaluated configurations: adding CLIP raises Recall@5 from 0.5250 to
0.6050 while citation F1 is lower, **0.2685 versus 0.2153**. The configurations
also differ in how much figure evidence reaches the generator, and figure
evidence carries weaker textual representation — a plausible account the frozen
experiments do not isolate.

A single blended "quality" number would have hidden this entirely. Retrieval
quality and citation behaviour are separate optimization problems and are measured
separately.

Citation metrics measure citation *behaviour*. They are not semantic
groundedness and not answer correctness —
[ADR-011](ADR-011-citation-aware-generation.md) decision 8.

### 9. Preserve negative results and frozen ablations as architectural evidence

The PR4 collapse is retained in `results/v2/`, in the ablation matrix, and in this
ADR. It is the reason the architecture has a reranking stage at all.

Deleting or overwriting a negative result would remove the justification for the
design that replaced it, leaving the architecture asserted rather than evidenced.
Configurations are frozen *before* results are inspected, so the matrix is a
measurement rather than a selection.

### 10. Treat reproducibility, provenance, and regression detection as architecture

Evidence provenance, deterministic evaluation, frozen baselines, and the two-tier
CI contract are parts of the system, not auxiliary tooling.

The retrieval contract — Recall@5 0.6050, MRR@5 0.5153, Figure Recall@5 0.4800,
with absolute tolerances 0.02/0.02/0.03 — is enforced on `main`
([ADR-012](ADR-012-two-tier-ci-quality-gates.md)). These are engineering
tolerances, not statistical significance.

## Consequences

**Positive**

- Complementary modality coverage: `retrieval_miss` down to 7/100 in the
  strongest evaluated frozen v2 configuration.
- Reproducible fusion: no tuned weights, deterministic tie-breaking, fixed `k`.
- The strongest evaluated frozen v2 configuration improves on every fusion-only
  configuration measured (Recall@5 0.6050 vs 0.5150 for the best RRF-only).
- Interpretable ablations: ten frozen configurations under identical conditions.
- Auditable evolution: every architectural claim links to a committed artifact.
- A measurable regression contract, verified to reproduce on two platforms.

**Negative / trade-offs**

- The cross-encoder is **text-only**. It cannot inspect pixels, so figure ranking
  depends entirely on textual representation — the weakest link (decision 7).
- Reranking's text recovery was **not free**: figure evidence moved down on
  average (mean rank 5.75 → 7.65) and 14 figure targets were pushed out of the
  top-5.
- Equal-weight RRF can produce modality competition on a dense figure universe
  (§15 of the architecture document).
- Additional latency and model footprint: 11.69 s one-off model load, ~43.9 ms
  mean per query warm, measured locally on macOS arm64.
- Hybrid both-target coverage@5 (0.32) still trails the Text+Caption RRF value
  (0.36) despite a large aggregate Recall@5 gain.
- Frozen v2 is small and informed development; every number is benchmark-specific.

## Alternatives considered

**Naive pooled ranking across modalities.** Rejected on principle, not on
measurement: it requires comparing a 768-d Ollama cosine against a 512-d CLIP
cosine, which have no common scale (decision 2). *Not evaluated in the frozen v2
experiments.*

**Caption-only retrieval.** Evaluated: Recall@5 0.3300, MRR@5 0.2835, Figure
Recall@5 0.5467. Strong on figures, much weaker overall than fusion.

**CLIP-only retrieval.** Evaluated: Recall@5 0.2900, MRR@5 0.1983. Weakest
single stream on ranking, yet contributes reachable candidates in combination —
which is precisely why single-stream results are a poor basis for dropping a
stream.

**RRF without a cross-encoder.** Evaluated: the three-stream configuration
collapsed the text slice to MRR@5 0.0000 and hybrid both-target coverage to 0.00.
This is the configuration the architecture moved away from.

**Dropping CLIP after PR4.** Would have been defensible on PR4 evidence alone and
would have cost the +0.0450 MRR@5, +0.0800 Recall@5 and +0.1067 Figure Recall@5
that reranking later unlocked. Retained as the clearest example of why a stream's
value cannot be judged independently of the stage that ranks it.

**Direct score blending with learned or hand-set weights.** *Not evaluated in the
frozen v2 experiments.* Deliberately avoided: tuning weights against a
100-query benchmark that informed development would produce a number without
generalization evidence.

**Multimodal (image-text) reranking.** *Not evaluated in the frozen v2
experiments.* It is the most promising route to the figure-representation
bottleneck of decision 7 and is listed under future research work. Its absence
here is a scope boundary, not an empirical rejection.

## Related ADRs

- [ADR-007 — Cross-Encoder Reranking](ADR-007-cross-encoder-reranking.md)
- [ADR-008 — Multimodal RAG Architecture](ADR-008-multimodal-rag-architecture.md)
- [ADR-009 — Canonical Retrieval in Production](ADR-009-canonical-retrieval-production-integration.md)
- [ADR-010 — Ablation Framework and Generation Evaluation](ADR-010-ablation-and-generation-evaluation.md)
- [ADR-011 — Citation-Aware Generation](ADR-011-citation-aware-generation.md)
- [ADR-012 — Two-Tier CI Quality Gates](ADR-012-two-tier-ci-quality-gates.md)
