# ADR-001 — src/ Layout and Library Design

**Status:** Accepted  
**Date:** 2026-06-04

---

## Context

The initial repo used a flat `app/` directory containing ingestion, retrieval, and API code alongside Streamlit entry points. This made the project hard to install as a Python package, mixed library code with application entry points, and created import path confusion when running tests or notebooks from different working directories.

The project aims to be:

1. A pip-installable Python library (`mrta`) for reuse across notebooks, scripts, and apps.
2. A portfolio project demonstrating senior-engineer hygiene (editable installs, typed config, CI).
3. A teaching resource where notebooks import from `mrta.*` instead of relative paths.

## Decision

Adopt the `src/` layout standard (PEP 517/518):

```text
src/
└── mrta/
    ├── core/          # config, schemas, llm client, rag pipeline
    ├── ingestion/     # pdf_loader, chunker, figure_extractor
    ├── retrieval/     # vector_store, embedder, reranker
    ├── multimodal/    # vlm_client, clip_embedder
    ├── evaluation/    # eval_pipeline, metrics
    ├── prompts/       # Jinja2 templates
    └── observability/ # logging, tracing

apps/
├── api/              # FastAPI entry point (imports from mrta.*)
└── streamlit/        # Streamlit UI entry point
```

The `src/` prefix prevents accidental imports from the working directory (a common subtle bug in flat layouts). `apps/` holds framework-specific entry points that are never imported by library code — keeping `src/mrta/` framework-agnostic.

Build backend: `hatchling` via `pyproject.toml`. Install with `pip install -e ".[all]"`.

## Consequences

**Positive:**

- Notebooks import cleanly: `from mrta.core.config import settings`
- Library is installable; entry points are decoupled from library logic.
- Tests import library code without path manipulation.
- No FastAPI or Streamlit in `src/mrta/` — library is framework-agnostic.

**Negative / Tradeoffs:**

- Slightly more directory depth than a flat layout.
- Contributors must `pip install -e .` before running notebooks (one-time setup).

## References

- [Python Packaging User Guide — src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [Hatchling build backend](https://hatch.pypa.io/latest/)
