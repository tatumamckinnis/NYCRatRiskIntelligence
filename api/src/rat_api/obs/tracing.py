"""OpenInference tracing setup (T-40).

Configures a TracerProvider with two BatchSpanProcessors:
  1. OTLPSpanExporter → Phoenix (or any OTLP collector) — no-op if endpoint is unset.
  2. JsonlSpanExporter → JSONL file sink at OBS_JSONL_PATH.

Span kind helpers set ``openinference.span.kind`` per the OpenInference spec so
Phoenix can classify spans as CHAIN / LLM / RETRIEVER / RERANKER.

Usage::

    from rat_api.obs.tracing import setup_tracing, get_tracer, llm_span

    setup_tracing("rat-api")

    tracer = get_tracer(__name__)
    with llm_span("generate"):
        ...
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

log = logging.getLogger(__name__)

# OpenInference span kind attribute key
_OI_SPAN_KIND = "openinference.span.kind"


# ---------------------------------------------------------------------------
# JSONL exporter
# ---------------------------------------------------------------------------

class JsonlSpanExporter(SpanExporter):
    """Writes one flattened JSON object per span to a JSONL file.

    All span attributes are merged into the top-level dict. This makes the
    file easy to parse with ``jq`` and read by the eval runner.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: object) -> SpanExportResult:  # type: ignore[override]
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                for span in spans:  # type: ignore[union-attr]
                    attrs = dict(span.attributes or {})
                    obj = {
                        "trace_id": format(span.context.trace_id, "032x"),
                        "span_id": format(span.context.span_id, "016x"),
                        "parent_span_id": (
                            format(span.parent.span_id, "016x") if span.parent else None
                        ),
                        "name": span.name,
                        "start_time_ns": span.start_time,
                        "end_time_ns": span.end_time,
                        "status": span.status.status_code.name,
                        **attrs,
                    }
                    fh.write(json.dumps(obj) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("JsonlSpanExporter: export failed: %s", exc)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------

_tracer_provider: TracerProvider | None = None


def setup_tracing(service_name: str, *, otel_endpoint: str = "", jsonl_path: str = "obs/traces.jsonl") -> None:
    """Configure the global TracerProvider.

    Args:
        service_name:   Logical service name shown in Phoenix.
        otel_endpoint:  OTLP HTTP/gRPC collector endpoint.  No-op if empty.
        jsonl_path:     Path to the JSONL span sink file.
    """
    global _tracer_provider  # noqa: PLW0603

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # --- OTLP exporter (Phoenix / Arize) ---
    if otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
                OTLPSpanExporter,
            )

            otlp = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp))
            log.info("OTLP tracing → %s", otel_endpoint)
        except Exception as exc:  # noqa: BLE001
            log.warning("OTLP exporter setup failed (continuing without it): %s", exc)

    # --- JSONL file sink ---
    jsonl = JsonlSpanExporter(jsonl_path)
    provider.add_span_processor(BatchSpanProcessor(jsonl))
    log.info("JSONL trace sink → %s", jsonl_path)

    trace.set_tracer_provider(provider)
    _tracer_provider = provider


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer for *name* from the configured provider."""
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Span-kind context managers
# ---------------------------------------------------------------------------

@contextmanager
def _span(kind_value: str, name: str) -> Generator[trace.Span, None, None]:
    tracer = get_tracer("rat_api")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(_OI_SPAN_KIND, kind_value)
        yield span


def chain_span(name: str) -> "contextlib.AbstractContextManager[trace.Span]":
    """Root orchestration span (CHAIN)."""
    return _span("CHAIN", name)


def llm_span(name: str) -> "contextlib.AbstractContextManager[trace.Span]":
    """LLM generation span."""
    return _span("LLM", name)


def retriever_span(name: str) -> "contextlib.AbstractContextManager[trace.Span]":
    """Vector / BM25 retrieval span."""
    return _span("RETRIEVER", name)


def reranker_span(name: str) -> "contextlib.AbstractContextManager[trace.Span]":
    """Re-ranking span (BGE Reranker or Cohere)."""
    return _span("RERANKER", name)
