"""Unit tests for mrta.observability.tracing.

Uses InMemorySpanExporter from the OTEL SDK to capture spans without any
external backend. The autouse fixture resets _configured and the global
TracerProvider between tests to keep each test fully isolated.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mrta.observability.tracing as tracing_module
from mrta.observability.tracing import configure_tracer, get_tracer, trace_span


@pytest.fixture(autouse=True)
def reset_tracing(monkeypatch):
    """Reset _configured and the global TracerProvider between tests.

    OTEL 1.25+ uses a Once guard (_TRACER_PROVIDER_SET_ONCE) that prevents
    set_tracer_provider from being called more than once. We reset both the
    guard's _done flag and _TRACER_PROVIDER so each test gets a clean slate.
    """
    monkeypatch.setattr(tracing_module, "_configured", False)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(trace._TRACER_PROVIDER_SET_ONCE, "_done", False)
    yield


@pytest.fixture
def in_memory_tracer() -> InMemorySpanExporter:
    """Install a TracerProvider backed by InMemorySpanExporter; return the exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


def test_get_tracer_returns_tracer() -> None:
    t = get_tracer()
    assert t is not None


def test_trace_span_noop_when_not_configured() -> None:
    with trace_span("mrta.noop"):
        pass  # must not raise when SDK not configured


def test_trace_span_yields_span(in_memory_tracer: InMemorySpanExporter) -> None:
    with trace_span("mrta.test") as span:
        assert span is not None


def test_trace_span_sets_attributes(in_memory_tracer: InMemorySpanExporter) -> None:
    with trace_span("mrta.attrs", {"query.length": 42, "model.llm": "llama3"}):
        pass
    spans = in_memory_tracer.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["query.length"] == 42
    assert spans[0].attributes["model.llm"] == "llama3"


def test_trace_span_records_exception(in_memory_tracer: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError):
        with trace_span("mrta.exc"):
            raise ValueError("boom")
    spans = in_memory_tracer.get_finished_spans()
    assert len(spans) == 1
    assert any(e.name == "exception" for e in spans[0].events)


def test_spans_captured_by_in_memory_exporter(in_memory_tracer: InMemorySpanExporter) -> None:
    with trace_span("mrta.rag_query"):
        pass
    names = [s.name for s in in_memory_tracer.get_finished_spans()]
    assert "mrta.rag_query" in names


def test_configure_tracer_sets_configured_flag() -> None:
    configure_tracer(console=False)
    assert tracing_module._configured is True


def test_configure_tracer_idempotent() -> None:
    configure_tracer(console=False)
    provider_after_first = trace.get_tracer_provider()
    configure_tracer(console=False)
    assert tracing_module._configured is True
    assert trace.get_tracer_provider() is provider_after_first


def test_configure_tracer_console_does_not_crash() -> None:
    configure_tracer(console=True)
    with trace_span("mrta.console_test"):
        pass


def test_span_name_preserved(in_memory_tracer: InMemorySpanExporter) -> None:
    with trace_span("mrta.ingestion"):
        pass
    assert in_memory_tracer.get_finished_spans()[0].name == "mrta.ingestion"
