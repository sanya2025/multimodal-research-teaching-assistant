# ADR-007: OpenTelemetry Tracing

**Status:** Accepted
**Date:** 2026-06-12

---

## Context

MRTA's RAG pipeline has three latency-contributing stages — vector search, reranking, and
LLM generation — but the existing observability layer (JSONL structured logging) only
captures end-to-end latency. Debugging slow queries or measuring which stage regresses
requires per-stage instrumentation.

The project also needs to remain deployable without any external backend — CI, local dev,
and minimal production environments should not depend on a tracing server.

---

## Decision

Adopt **OpenTelemetry** as the tracing standard for MRTA.

- `opentelemetry-api` is added to base dependencies. It provides a no-op proxy tracer
  when no SDK is configured — importing and calling `get_tracer()` has zero cost without
  an SDK.
- `opentelemetry-sdk` is added to the `[dev]` and `[otel]` optional groups. It is only
  activated when `configure_tracer()` is called at process startup.
- `opentelemetry-exporter-otlp-proto-grpc` is in `[otel]` only, for production OTLP export.
- Tracing is controlled by `enable_tracing: bool = False` in `Settings`. All instrumented
  code paths are no-ops when the SDK is not configured.

---

## Alternatives considered

| Alternative | Rejected because |
|-------------|-----------------|
| **structlog-only** | Requires parsing log lines to reconstruct stage timings; not queryable across requests |
| **Custom timing dict** | Non-standard; no tooling for visualization or alerting; reinvents spans |
| **Jaeger Python client (direct)** | Vendor lock-in; deprecated in favor of OTEL |
| **Datadog / Honeycomb SDK** | SaaS lock-in; paid; violates the "works locally" requirement |
| **OpenTelemetry as a required dep** | `opentelemetry-sdk` is non-trivial; should not be forced on users who don't want tracing |

---

## Consequences

**Positive:**
- Vendor-neutral: spans can be exported to Jaeger, Grafana Tempo, Zipkin, or any
  OTLP-compatible backend.
- Truly optional: `pip install mrta` (no extras) has no tracing overhead beyond the
  `opentelemetry-api` proxy (pure Python, ~200 KB).
- Industry standard: reduces friction for teams already using OTEL.
- Per-stage latency now visible without adding debug prints.

**Negative:**
- `opentelemetry-api` is now a base dependency. It is lightweight and widely used, but
  it does increase the install footprint slightly.
- `configure_tracer()` must be called once at startup — forgetting this means tracing
  silently produces no-ops even when `enable_tracing=True`. The README documents this.
- `_configured` guard is a module-level bool that must be reset in tests using
  `monkeypatch` — a minor test-isolation constraint.

---

## Implementation

- `src/mrta/observability/tracing.py` — `configure_tracer()`, `get_tracer()`, `trace_span()`
- Instrumented: `rag_pipeline.rag_query()`, `chunker.chunk_pdf()`, `eval_pipeline.run_eval()`
- Tests: `tests/unit/test_tracing.py` using `InMemorySpanExporter`
- Documentation: `docs/observability.md`
