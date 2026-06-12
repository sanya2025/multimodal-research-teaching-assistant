# Observability

MRTA ships two complementary observability layers: structured logging (JSONL per run) and
optional OpenTelemetry distributed tracing.

---

## Structured logging

Every RAG run can be logged to a JSONL file via `StructuredLogger.log_run()`. The output
path is controlled by the `LOG_FILE` setting (default: `data/logs/runs.jsonl`).

---

## Distributed tracing (OpenTelemetry)

Tracing is **disabled by default**. The project works without any OTEL backend installed.
When enabled, spans are emitted for each stage of the RAG lifecycle.

### What is traced

| Span name | Where | Key attributes |
|-----------|-------|----------------|
| `mrta.rag_query` | `rag_pipeline.rag_query()` | `query.length`, `retrieval.top_k`, `reranker.enabled`, `reranker.top_n`, `retrieval.chunk_count`, `model.llm`, `latency_ms` |
| `mrta.ingestion` | `chunker.chunk_pdf()` | `document.path`, `chunk.strategy`, `chunk.count` |
| `mrta.evaluation` | `eval_pipeline.run_eval()` | `benchmark.size` |

### Enable console tracing (local dev)

Set these environment variables or add them to your `.env` file:

```bash
ENABLE_TRACING=true
OTEL_CONSOLE_EXPORTER=true
```

Then wire up the tracer at process startup (e.g., in a script or `__main__`):

```python
from mrta.core.config import settings
from mrta.observability.tracing import configure_tracer

if settings.enable_tracing:
    configure_tracer(
        service_name=settings.otel_service_name,
        console=settings.otel_console_exporter,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
    )
```

Spans will print to stdout in JSON format as each RAG call completes.

### Configure an OTLP endpoint (Jaeger, Tempo, etc.)

Install the OTLP exporter group:

```bash
pip install -e ".[otel]"
```

Set the endpoint:

```bash
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=mrta
```

Start a local Jaeger instance:

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

Open `http://localhost:16686` to view traces after running queries.

### Useful span attributes for RAG debugging

| Attribute | What to look for |
|-----------|-----------------|
| `latency_ms` | High values → slow LLM or vector search |
| `retrieval.chunk_count` | Lower than `top_k` → index is sparse |
| `reranker.enabled` | Confirm reranking is active when expected |
| `model.llm` | Confirm the correct model is being used |
| `chunk.strategy` | Verify ingestion strategy matches config |

### Why tracing helps for RAG

A RAG query has three latency-contributing stages: vector search, optional reranking, and
LLM generation. Without tracing, latency is opaque — a slow response could be any of the
three. Spans make each stage's contribution visible and make regressions diagnosable without
adding debug prints to production code.
