# ADR-015 — Make Per-Stream Degradation Observable

**Date:** 2026-09-25
**Status:** Proposed
**Branch:** _not yet implemented — proposed on `fix/visual-evidence-serving-path`_
**Relates to:**
[ADR-007](ADR-007-opentelemetry-tracing.md),
[ADR-009](ADR-009-canonical-retrieval-production-integration.md),
[ADR-012](ADR-012-two-tier-ci-quality-gates.md),
[ADR-014](ADR-014-visual-evidence-serving-path.md)

---

## Context

[ADR-014](ADR-014-visual-evidence-serving-path.md) repaired four defects in the
visual evidence path and closed with one cost it deliberately did not pay:

> **Three swallowed exceptions are now understood but not removed.** Per-stream
> degradation is still the right default — a missing caption index should not fail
> a query. The cost is that a wiring error is indistinguishable from an absent
> optional component.

This ADR pays that cost. It is a reporting change, not a policy change: nothing
here makes a degraded stream fail a request.

The reason ADR-014's Defect 1 survived long enough to reach a user is worth
stating precisely, because it was not a missing `try`/`except`. Every layer that
detected the failure either recorded it and never reported it, or reported a
value that was true.

### Three degradation surfaces exist, with three different fates

**Ingestion already records reasons, and they never leave the process.**
`IndexResult.degraded: dict[str, str]`
([`document_indexer.py:92`](../../src/mrta/ingestion/document_indexer.py#L92))
stores `f"{type(exc).__name__}: {exc}"` for `figure_extraction`, `caption_index`
and `clip_index`. The data is well-formed and specific. But
[`UploadResponse`](../../apps/api/schemas/upload.py) has no field for it, and
`/upload` constructs its response from the counts only. A caller cannot observe
that a stream failed, and neither can a developer without attaching a debugger.

**Query time records reasons, then discards them at the only boundary that
reports them.** `RetrievalDiagnostics.degraded_streams: dict[str, str]`
([`canonical_pipeline.py:169`](../../src/mrta/retrieval/canonical_pipeline.py#L169))
captures caption, CLIP and reranker failures with the same reason strings. Those
reach exactly one consumer, an OpenTelemetry span attribute — and it is written as
`sorted(diagnostics.degraded_streams)`
([`canonical_pipeline.py:354`](../../src/mrta/retrieval/canonical_pipeline.py#L354),
[`canonical_rag.py:341`](../../src/mrta/generation/canonical_rag.py#L341)).
Sorting a dict iterates its **keys**, so the span records *which* streams degraded
and throws away *why*. The cause was collected and then dropped one line before
it became visible.

**Startup records nothing at all.** `_warm_torch_runtime()` returns `None` when
CLIP weights cannot load, and the lifespan's `except Exception` sets
`app.state.retriever = None` and `app.state.vlm = None`
([`main.py`](../../apps/api/main.py)). Neither writes down what happened.
`/health` ([`main.py:265`](../../apps/api/main.py#L265)) returns a static
`{"status": "ok"}` — it is a liveness probe that cannot distinguish a fully
multimodal server from one serving text only.

### The one readiness field `/upload` does report was the field that lied

`IndexResult.visual_retrieval_available` is defined as
`n_caption_records > 0 or n_visual_records > 0`
([`document_indexer.py:94-97`](../../src/mrta/ingestion/document_indexer.py#L94-L97)).

Under ADR-014 Defect 1 the caption stream worked and the CLIP stream raised on
every call. `n_caption_records > 0` was therefore true, so the **disjunction
returned `True`** while half the visual system was inert. The single boolean that
existed to answer "is visual retrieval available?" answered yes, correctly by its
own definition and uselessly in practice. An `or` across independent streams
cannot express partial failure.

### What the system cannot currently say

There are three distinct states, and the codebase has vocabulary for two of them:

| State | Meaning | Currently expressible |
|---|---|---|
| `ready` | Component loaded and serving | Yes, implicitly (not `None`) |
| `disabled` | Absent by configuration or missing extra — expected | No — looks identical to `failed` |
| `failed` | Configured and expected, but raised at runtime | No — looks identical to `disabled` |

Collapsing `disabled` and `failed` is the whole defect. `enable_canonical_retrieval=false`
and "CLIP raised `AttributeError` on every figure" produce the same observable
system: a `None` and a silent absence of visual results.

## Decisions

### 1. Three-state component status, not a boolean

Every optional component reports `ready`, `disabled` or `failed`, and `failed`
carries the reason string that is already being constructed today. This is the
minimum vocabulary needed to tell a configuration choice from a bug, and no
existing field can be widened to carry it.

### 2. `/health` reports component readiness additively, and its HTTP status never changes

`/health` gains a `components` object. The existing `status: "ok"` key keeps its
current meaning and value, and the endpoint keeps returning **200 whenever the
process is alive**, including when every optional component has failed.

This constraint is load-bearing, not conservatism. `compose.yaml` probes `/health`
with a bare `urlopen` and gates Streamlit behind `depends_on: service_healthy`.
A `/health` that went unhealthy on an optional-stream failure would stop the UI
from starting at all — converting a degraded visual stream into a total outage of
the text path that still worked. ADR-014 already had to raise `start_period` to
180s after a healthcheck mismatch blocked Streamlit; this ADR must not reintroduce
that class of trap in a worse form.

Liveness and readiness stay separate: the status code answers "is the process
alive", the body answers "what is it able to do".

### 3. Startup degradation gets a recorder

`app.state.component_status` is populated by the lifespan as it wires each
component — including the two paths that currently discard their failure,
`_warm_torch_runtime()` returning `None` and the multimodal `except Exception`.
`/health` reads this structure rather than re-deriving readiness from `None`
checks, because a `None` cannot say why it is `None`.

### 4. `/upload` returns the `degraded` map ingestion already produces

`UploadResponse` gains a `degraded: dict[str, str]` field, defaulting to `{}` and
populated from `IndexResult.degraded`. No new data is computed — this is a field
addition on a response, additive in the same sense as the PR6 counts, so existing
clients are unaffected.

An ingestion that silently reported success while one index failed to build is
precisely ADR-014 Defect 1, and this is the smallest change that would have made
it visible at the moment it happened.

### 5. `visual_retrieval_available` keeps its definition and gains company

The boolean is **not** redefined. It is a PR6 response contract, and tightening it
to require both streams would change the meaning of an existing field for existing
callers — a silent behaviour change of the kind ADR-014 was written to stop.
Instead, per-stream status appears alongside it, and the boolean's disjunction
semantics are documented as deliberate in its docstring.

### 6. Span attributes carry reasons, not only stream names

`sorted(diagnostics.degraded_streams)` becomes an attribute pair that preserves the
reason strings. ADR-007 established tracing as the diagnostic channel for
per-request behaviour; a channel that records that something failed but not what
failed is not discharging that decision.

### 7. Degradation policy is unchanged

Per-stream graceful degradation (ADR-009 §6) stays exactly as it is. No handler is
removed, no optional stream becomes required, and no request that succeeds today
begins to fail. The `noqa: BLE001` comments stay, because the broad catches are
still correct — they were never the bug. The bug was that nothing downstream could
read what they caught.

### 8. No new dependency and no metrics backend

Reporting rides on what already exists: response fields and OpenTelemetry spans.
Introducing a metrics server or time-series backend for a single-node, local-first
project would add operational weight out of proportion to the problem.

## Consequences

**ADR-009 §7 becomes verifiable at runtime.** "Production ingestion builds all
three indices" was true as a decision and false as a fact for the CLIP index, and
the gap was only findable by reading code. After this change a single `/upload`
response or `/health` body answers it.

**A class of bug becomes loud rather than impossible.** This ADR does not prevent
wiring errors — Decision 1 in ADR-014 (one encoder per role) is what addresses the
specific cause. What changes is the time-to-detection: from months to the first
request. That is the honest claim, and it is smaller than "this prevents
regressions".

**`/health` acquires a body contract.** Once clients read `components`, its shape
is a compatibility surface. It is additive on introduction, but future changes to
it are breaking changes in a way that today's `{"status": "ok"}` is not.

**Startup cost is unchanged.** Recording status is bookkeeping around work the
lifespan already performs; it adds no model load and no I/O.

**Test surface grows in Tier 1.** Component status is pure structure with no model
dependency, so it tests as fast unit tests under `MRTA_ENV=test` and needs no
`heavy` marker (ADR-012).

**This adds no alerting.** Nothing watches `/health` and nothing pages. The
change makes state observable to a human or a client that looks; it does not make
anything notice on its own.

## Alternatives considered

**Fail closed — a stream that raises fails the request.** Rejected: it directly
contradicts ADR-009 §6 and inverts the cost. Under Defect 1 it would have turned a
partial visual regression into a complete outage of a working text path. A missing
optional index must not fail a query; that decision is sound and is not reopened.

**Log the reasons and stop there.** Rejected: this is effectively the status quo
and it is what failed. The reasons *were* recorded, as structured strings, for
months. Recording without a reader is the defect, and adding a log line does not
add a reader.

**Export Prometheus metrics.** Rejected for now: it is the right answer for a
multi-instance deployment and the wrong weight for this one. Nothing in the
project runs a scrape target, and ADR-003's local-first posture means a metrics
backend would be infrastructure that exists only to serve this ADR.

**Return 503 from `/health` when a component failed.** Rejected as actively
harmful — see Decision 2. It would deadlock `depends_on: service_healthy` and take
the UI down over an optional stream.

**Make ingestion raise when a configured index fails to build.** Tempting, and
narrower than fail-closed at query time: an upload that cannot build a configured
index arguably did not succeed. Deferred rather than rejected — it changes
`/upload`'s success semantics and deserves its own evidence about partial-upload
recovery, which this ADR does not gather.

## Open questions

1. **Should Streamlit surface degradation?** A user asking a visual question of a
   text-only server currently gets a confident answer with no figures and no
   explanation. Showing component status in the UI is a product decision, not a
   plumbing one.
2. **Should the evaluation harness assert component readiness?** A benchmark run
   against a silently degraded stack would produce numbers that look valid.
   ADR-012's configuration-identity check guards the models but not stream
   liveness.
3. **Does `retrieval_mode="multimodal"` remain honest when the CLIP stream is
   `failed`?** ADR-014 Defect 3 made the label truthful about images; the label
   may still overstate which streams were actually consulted.

## Related ADRs

- [ADR-007 — OpenTelemetry Tracing](ADR-007-opentelemetry-tracing.md)
- [ADR-009 — Canonical Retrieval in the Production Query Path](ADR-009-canonical-retrieval-production-integration.md)
  — §6 degradation policy, §7 ingestion claim
- [ADR-012 — Two-Tier CI Quality Gates](ADR-012-two-tier-ci-quality-gates.md)
- [ADR-014 — Visual Evidence Serving Path](ADR-014-visual-evidence-serving-path.md)
  — names this as the deliberate follow-up
