"""mrta.observability.tracing — optional OpenTelemetry instrumentation."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Span

_configured: bool = False


def configure_tracer(
    service_name: str = "mrta",
    console: bool = False,
    otlp_endpoint: str = "",
) -> None:
    """Set up the OTEL SDK once. Call at process startup when enable_tracing=True.

    console=True logs spans to stdout (useful for local dev).
    otlp_endpoint exports spans via OTLP/gRPC to Jaeger, Tempo, etc.
    """
    global _configured
    if _configured:
        return
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    if console:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str = "mrta") -> trace.Tracer:
    """Return the module-level tracer. No-op proxy when SDK not configured."""
    return trace.get_tracer(name)


@contextmanager
def trace_span(
    span_name: str,
    attributes: dict | None = None,
) -> Generator[Span, None, None]:
    """Context manager that wraps a block in a named span.

    Attributes are set after span creation. Exceptions are recorded and re-raised.
    No-op when tracing is not configured — safe to leave in production code.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise


__all__ = ["configure_tracer", "get_tracer", "trace_span"]
