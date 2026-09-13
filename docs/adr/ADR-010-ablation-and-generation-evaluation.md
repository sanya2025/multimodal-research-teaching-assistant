# ADR-010 — Ablation Framework and Generation Evaluation

**Date:** 2026-09-12
**Status:** Accepted
**Branch:** `feat/eval-ablation-runner-generation-metrics` (PR7)
**Relates to:** [ADR-006](ADR-006-evaluation-framework.md), [ADR-007](ADR-007-cross-encoder-reranking.md), [ADR-009](ADR-009-canonical-retrieval-production-integration.md)

---

## Context

PR1–PR6 produced a working multimodal stack and a frozen 100-query benchmark,
but each stage's contribution was measured by a bespoke script written for that
PR. Comparing PR2's caption numbers against PR5's reranked numbers meant
comparing two programs, and nothing prevented a later change from silently
altering an earlier result.

PR7 needed one framework that runs every configuration under identical
conditions, adds generation-quality measurement, and can attribute a failure to
the stage that caused it.

## Decisions

### 1. Configuration matrix frozen before results were seen

Nine retrieval configurations plus one oracle, declared in
`FROZEN_CONFIGURATIONS` and mirrored in `configs/ablation_config.yaml` (a test
asserts the two cannot drift). Declaring them up front is what makes the
comparison an ablation rather than a search: adding a configuration after seeing
results and reporting it alongside the rest would silently convert measurement
into selection.

Invalid combinations are rejected rather than silently ignored — a single stream
cannot be "fused", and reranking requires a fused pool, because PR5 reranks the
RRF pool and not a raw stream.

### 2. Evaluation uses `EvalAdapter`, never the production adapter

The two disagree on `document_id` by design: production keys documents by the
content-hashed `doc_id`, benchmark targets by the manifest id. This is not a
detail — wiring the production adapter made every metric read 0.0000 while
retrieval was working perfectly, because nothing ever matched.

The runner now requires an adapter and fails loudly without one.

### 3. Figure representation follows PR5's frozen policy exactly

A `VisualRecord` from the CLIP index carries no text, so a CLIP-only figure would
otherwise reach the cross-encoder as an empty string. Both visual streams take
figure text from one canonical-id lookup built over the whole caption index, and
multi-crop figures resolve deterministically (prefer a VLM caption, then lowest
`evidence_id`) rather than by whichever crop the query happened to return.

Without both rules the reranked configurations drifted from their frozen values.
Reproducing history is what caught it.

### 4. Nothing is called "faithfulness"

Every grounding signal in `generation_metrics.py` is lexical. The names say
`lexical_support_score`, `supported_claim_fraction`, `unsupported_claim_fraction`
because that is what they measure: token overlap with supplied context, which
underestimates correct paraphrase and overestimates coincidental vocabulary.

A sentence counts as supported only when at least half its content tokens appear
in the context. The obvious alternative — "supported if *any* token appears",
which is what `mrta.evaluation.metrics.faithfulness` does — marks nearly
everything supported and carries almost no signal.

The Stage-7 functions keep their names because `EvalReport` and `run_eval` depend
on them, but they claim more than they measure and PR7 does not reuse them.

### 5. Validity, relevance and coverage stay separate

- **validity** — did the citation resolve to evidence the generator was given?
- **relevance** — was the cited evidence the *expected* evidence? (precision)
- **coverage** — was the expected evidence cited at all?

A structurally valid but irrelevant citation is a precision loss, not a
hallucination. Collapsing these into one "citation accuracy" number would hide
which stage is at fault, which is the ablation's entire purpose.

### 6. Failure attributed to the earliest failing stage

`retrieval_miss` → `ranking_miss` → `citation_validity` → `citation_missing` →
`citation_relevance` → `answer_support`. If the target never reached the
generator, a missing citation is not the generator's fault; attributing it there
would make generation look worse than it is.

### 7. Two provenance groupings, never conflated

`target_figure_provenance` groups by the representation of the figure the
benchmark *expects* — PR5's grouping, and the only one comparable with its
measurements. `top_retrieved_figure_provenance` groups by the first figure the
system actually surfaced.

They describe different populations. An earlier draft reported only the second
under the bare name `figure_provenance`, which read as though it contradicted
PR5's VLM-caption-vs-fallback finding when it was simply measuring something
else. Both are now emitted, and both are named for what they measure.

### 8. Citation recall is not answer correctness

Citation recall measures *citation completeness*: how much of the expected
evidence the answer pointed at. An answer can be correct while citing one of two
redundant targets, and it would score 0.5 here.

Nothing in this module measures semantic correctness. Reports must say "did not
cite every expected evidence item", never "was incorrect" or "hallucinated".

### 9. The oracle receives real evidence text

`oracle_evidence_generation` hands the generator the ground-truth evidence
resolved to actual text from the same indices retrieval reads — text targets to
their page's chunks, figure targets to the canonical figure text. An oracle fed
bare identifiers measures empty-context generation, not the generation ceiling.
Its retrieval metrics are perfect by construction and must never be mixed into a
retrieval comparison.

### 10. Deterministic by default, generation opt-in

Retrieval-only is the default, so the standard run needs no Ollama and no model
download. Generation runs only under `--generation`, pinned to `temperature=0`
evaluation-side (not a production default change). Unit tests use fakes
throughout.

Failed queries are counted separately and excluded from metric means, so an
execution error cannot improve an average by dropping a hard query from the
denominator.

## Consequences

**Positive**

- One framework replaces five bespoke scripts; every configuration runs under
  identical conditions.
- The historical-reproduction check is a standing regression test on PR1–PR5.
- Failure decomposition separates retrieval bottlenecks from generation ones.
- Retrieval pools are retrieved once per query and reused across all ten
  configurations, so a full retrieval sweep takes ~15s.

**Negative / tradeoffs**

- Grounding proxies are lexical and will misjudge paraphrase. A semantic judge is
  deliberately deferred.
- Token counts use `tiktoken cl100k_base`, not the Ollama generator's own
  tokenizer: comparable across configurations, but not the model's exact
  accounting. The method is recorded in every result.
- The oracle's text rendering is a reasonable reconstruction, not the exact
  context a perfect retriever would have produced.
- v2's 21 canonical figures against a depth-20 pool remain structurally
  saturated; PR7 measures that benchmark faithfully but cannot widen it.

## What the first run measured

Two findings shaped the conclusion more than the framework itself.

**Failure is distributed, not concentrated in one stage.** For `full_reranked`:
ranking_miss 36, citation_missing 28, citation_relevance 23, retrieval_miss 7.
Generation-side behaviour is the largest single category, but ranking remains a
material source of error, and only 7/100 queries fail for lack of retrievable
evidence. Oracle-evidence generation reaches only 0.605 citation recall —
perfect availability does not guarantee complete citation. None of this is a
claim about semantic correctness.

**Improving retrieval can degrade citation quality.** Adding CLIP raises R@5
from 0.525 to 0.605 while citation F1 falls from 0.269 to 0.215. It cuts
retrieval misses from 24 to 7 and nearly doubles the figure share of the
generation context (19.6% → 35.8%); that figure evidence carries weaker textual
representation, so citation precision drops from 0.756 to 0.596 on the queries
whose context changed.

The error relocates from retrieval to citation rather than disappearing. This is
a retrieval-to-generation *interface* problem: the system's weakest link is how
figure evidence is represented textually once retrieved, which is the same
weakness PR5 measured (VLM-caption targets FigR@5 0.703 vs nearby-text fallback
0.263, reproduced exactly here).

It argues for improving figure representation before investing in generation-side
citation behaviour, because better retrieval alone measurably makes citation
quality worse.

## Alternatives considered

**Extend the PR5 script.** Rejected: it hardcodes four configurations and its
comparison baselines, and would have to be rewritten for each new ablation.

**LLM-as-judge for faithfulness.** Deferred. It would make the deterministic
suite depend on a model and make regression runs non-reproducible. The metric API
leaves room for an optional judge later.

**Reuse `mrta.evaluation.metrics`.** Rejected on naming grounds — see decision 4.

## Related ADRs

- [ADR-006 — Evaluation Framework](ADR-006-evaluation-framework.md)
- [ADR-007 — Cross-Encoder Reranking](ADR-007-cross-encoder-reranking.md)
- [ADR-009 — Canonical Retrieval in Production](ADR-009-canonical-retrieval-production-integration.md)
