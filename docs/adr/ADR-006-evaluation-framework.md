# ADR-006 — Evaluation Framework

**Status:** Accepted  
**Date:** 2026-06-04

---

## Context

RAG systems are notoriously difficult to evaluate because quality is multidimensional: an answer can be fluent but hallucinated, grounded but irrelevant, or relevant but missing citations. A formal evaluation framework is needed to:

1. Detect regressions when chunking or retrieval parameters change.
2. Demonstrate evaluation methodology for portfolio purposes.
3. Provide CI-friendly assertions (pass/fail thresholds).

Frameworks evaluated: **Ragas**, **DeepEval**, manual annotation.

## Decision

**Primary: Ragas. Secondary: DeepEval for assertion-based CI checks.**

### Metrics tracked per question

| Metric | Tool | What it measures |
|---|---|---|
| Answer relevance | Ragas | Does the answer address the question? |
| Faithfulness / Groundedness | Ragas | Is every claim supported by retrieved context? |
| Context precision | Ragas | Fraction of retrieved chunks that were useful |
| Citation correctness | Custom | Do cited page numbers actually contain the claim? |
| Latency | Custom | End-to-end seconds (p50, p95) |
| Hallucination rate | DeepEval | Claims not present in any retrieved chunk |

### Benchmark dataset

A small hand-labeled benchmark (10–20 question/answer/source triples) over the "Attention Is All You Need" paper. Questions cover: factual recall, equation reference, figure description, multi-hop reasoning.

Defined in `tests/fixtures/` (Phase 4). Eval pipeline in `src/mrta/evaluation/eval_pipeline.py`.

### CI integration

DeepEval's assertion mode (`assert_test`) runs in `pytest` with hard thresholds:

- Faithfulness ≥ 0.7
- Answer relevance ≥ 0.6

These are conservative starting thresholds — tightened as the system matures.

## Consequences

**Positive:**

- Ragas provides LLM-based metrics without building custom judges.
- DeepEval assertions give CI a clear pass/fail signal.
- Citation correctness metric is custom — differentiates this project from generic RAG demos.

**Negative / Tradeoffs:**

- Ragas requires an LLM to score answers; adds latency to eval runs.
- Small benchmark (10–20 items) has high variance; not suitable for statistical significance claims.
- LLM-based metrics are themselves imperfect — documented limitation.

## References

- [Ragas documentation](https://docs.ragas.io/)
- [DeepEval documentation](https://docs.confident-ai.com/)
- [ARES — RAG evaluation paper](https://arxiv.org/abs/2311.09476)
- [Notebook 09 — Evaluation, Logging, Docker](../../notebooks/tutorials/2026-05-25-phase09-evaluation-logging-docker.ipynb)
