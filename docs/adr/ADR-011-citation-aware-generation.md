# ADR-011 — Citation-Aware Generation and Evidence Utilization

**Date:** 2026-09-13
**Status:** Accepted
**Branch:** `feat/generation-citation-aware-grounding` (PR8)
**Relates to:** [ADR-009](ADR-009-canonical-retrieval-production-integration.md), [ADR-010](ADR-010-ablation-and-generation-evaluation.md)

---

## Context

PR7 measured that failure is distributed across stages: in the full reranked
system only 7/100 queries fail for lack of retrievable evidence, 36 fail at
ranking, and a substantial further fraction fail after correct evidence reaches
the generator. Oracle-evidence generation reached only 0.605 citation recall.

That raised a question PR7 could not answer: is incomplete citation a property of
the generator, or of how evidence is *presented* to it? PR8 isolates the second
by holding retrieval completely fixed and varying only the prompt and the output
contract.

## Decisions

### 1. Retrieval is frozen, and the freeze is proven rather than asserted

Every condition receives the identical top-5 canonical evidence. An
``evidence_context_hash`` over each label's canonical identity and text is
recorded per row, and the run reports how many `(query, configuration)` pairs
supplied identical evidence to every condition. The final run: **200/200**.

Without this the experiment has no causal claim — a prompt difference and an
evidence difference would be indistinguishable in the results.

### 2. Four conditions, frozen before the final run

| Condition | Varies |
|---|---|
| `g0_baseline` | nothing — the unchanged PR6/PR7 production prompt |
| `g1_explicit_citations` | claim-level citation requirements added |
| `g2_structured_evidence` | evidence presented as typed cards |
| `g3_structured_generation` | cards plus a JSON `{answer, citations}` contract |

Template hashes were recorded before the comparative run. Prompt mechanics were
developed against synthetic unit-test examples, never against aggregate v2
outcomes.

### 3. The model selects labels; the application owns canonical identity

Unchanged from PR6, and extended to structured output. A G3 reply's `citations`
list is normalised through a strict label pattern: `"T1"`, `"[t1]"` and `" f2 "`
resolve; `"page 4"`, `"figure 2"` and `"/etc/passwd"` do not. The model cannot
inject a document id, page, figure id or path into a citation — it can only
choose among labels the application assigned.

### 4. Structured output degrades safely, and says so

Ollama's JSON mode constrains decoding but does not guarantee a parseable reply.
Every failure path — invalid JSON, a JSON array, a missing `answer`, a
non-list `citations` — falls back to inline label extraction and records
`malformed_structured_output`. Silently succeeding on a bad parse would convert
a generation failure into a citation failure and misattribute it.

### 5. Generation failures are a list, not a category

Unlike PR7's earliest-stage retrieval attribution, generation failures co-occur:
one answer can omit an expected citation *and* add an irrelevant one *and* leave
claims uncited. Collapsing that to one label discards most of the signal.

### 6. Aggregation is keyed by (query, configuration) — never by query alone

This is recorded as a decision because getting it wrong produced two wrong
results during PR8, one of which reversed a stated conclusion.

The oracle configuration supplies ground-truth evidence by design, so its rows
legitimately differ from a retrieval configuration's. Grouping by `query_id`
alone first produced a false integrity FAIL, then — more seriously — averaged
oracle rows into the headline condition table, inflating every condition and
inverting the apparent G0→G1 direction.

`_paired_analysis` now **raises** when handed rows from more than one
configuration rather than averaging them, and both grouping bugs have regression
tests.

### 7. G3 is a combined intervention, and is reported as one

G3 changes **two** things relative to G2: typed evidence cards *and* the JSON
`{answer, citations}` contract. Its improvement therefore demonstrates the effect
of that **combination**, not of structured output alone.

Separating the two would require a fourth cell — "G0 prompt + JSON output" —
which is absent from the frozen matrix. It is deliberately not being added:
introducing a condition after seeing results converts measurement into tuning,
which is precisely what freezing the matrix was for. The confound is recorded as
a limitation instead, and the factorial cell belongs in a future experiment
designed with it from the start.

### 8. Citation completeness is not semantic correctness

Nothing in PR8 measures whether an answer is *right*. Citation recall measures
how completely an answer pointed at the expected evidence. An answer can be
correct while citing one of two redundant targets, and would score 0.5.

`claim_citation_coverage` measures citation *behaviour* — the fraction of
sentences carrying a label — and says nothing about whether those sentences are
grounded or true.

### 9. No semantic judge, and no production change

A judge would make the deterministic suite depend on a model and make regression
runs non-reproducible; it is deferred. PR8 also changes no production behaviour:
the conditions are evaluation-only, so no feature flag was introduced. The one
production-code change is an optional `response_format` parameter on
`LLMClient.chat`, defaulting to None, which no existing caller passes.

## Consequences

**The headline is a trade-off, not a win.** With identical retrieved evidence,
structured citation generation raised citation recall 0.3100 → 0.4300 and
citation F1 0.2203 → 0.3603, while *reducing* precision 0.5883 → 0.3500. G3 moves
the system from under-citing toward over-citing.

That reframes the open question. It is no longer "can we get the model to cite?"
— it is "which citations actually support the claims, and are the resulting
answers semantically correct?" PR8's deterministic metrics cannot answer the
second, which is what makes a semantic judge the next step rather than more
prompt work.

**Positive**

- The generation/evidence interface is now measurable with retrieval held fixed.
- G3 improves citation F1 over the baseline on the real system (0.2203 → 0.3603)
  and claim citation coverage more than doubles (0.2650 → 0.6015).
- Structured output is also *faster* — 674 ms mean versus 1409 ms for G0 — because
  the JSON contract produces shorter replies.

- The visual result connects PR8 to the earlier representation findings without
  touching retrieval: G3 improves `visual_layout` citation F1 most of any intent
  slice (0.1150 → 0.3500) and `visual_caption` nearly as much (0.2167 → 0.4000),
  yet the PR5 representation gap persists — VLM-captioned target figures beat
  nearby-text fallbacks under *every* condition (G3: 0.4243 vs 0.2588). Better
  citation behaviour does not compensate for a weak figure description.

**Negative / tradeoffs**

- G3 trades precision for recall: 0.5883 → 0.3500. It cites more, and more of
  what it cites is wrong. Whether that trade is desirable depends on whether a
  reader is harmed more by a missing citation or a spurious one — a product
  question PR8 does not answer.
- Structured prompting costs ~30% more prompt tokens (940 → 1223).
- Every result is specific to one local generator at temperature 0.

## Alternatives considered

**Change the production generation path to G3.** Rejected for now: the
precision regression is large and unexplained, and PR8 measures citation
behaviour rather than answer quality.

**Add a semantic judge to resolve the precision question.** Deferred — it is the
natural next step, but it would change the character of the suite.

## Related ADRs

- [ADR-009 — Canonical Retrieval in Production](ADR-009-canonical-retrieval-production-integration.md)
- [ADR-010 — Ablation Framework and Generation Evaluation](ADR-010-ablation-and-generation-evaluation.md)
