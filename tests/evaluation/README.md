# Evaluation Tests

Deterministic regression gates for RAG retrieval quality. No LLM calls, no network
access, no model downloads — all tests run offline in CI.

## Purpose

The golden QA dataset (`datasets/golden_qa.yaml`) contains 30 hand-curated questions
with `expected_documents` (PDF filenames the retriever should surface) and
`expected_keywords`. These tests verify that metric functions are correct and that
retrieval quality does not regress below the stored baselines.

## Running the tests

```bash
pytest tests/evaluation -v
```

To run a single file:

```bash
pytest tests/evaluation/test_retrieval_metrics.py -v
pytest tests/evaluation/test_citation_coverage.py -v
pytest tests/evaluation/test_rag_gates.py -v
```

## Test files

| File | What it tests |
|------|--------------|
| `test_retrieval_metrics.py` | Pure unit tests for `recall_at_k`, `mrr`, `ndcg_at_k`, `citation_coverage` |
| `test_citation_coverage.py` | Citation coverage against golden QA items; loader correctness |
| `test_rag_gates.py` | Delta-based CI gates loaded from `baselines/retrieval_metrics.json` |

## Updating thresholds

Baselines live in `baselines/retrieval_metrics.json`. Each entry has two fields:

```json
"recall_at_5": { "baseline": 0.85, "tolerance": 0.05 }
```

- **`baseline`** — the expected score from the real retriever. Update this after
  connecting `VectorStore.search()` and measuring actual performance.
- **`tolerance`** — allowed regression before CI fails. Do not reduce this without
  deliberate discussion. The current value (0.05) absorbs small natural variation.

Gate assertion: `score >= baseline - tolerance`.

## Adding new questions

1. Open `datasets/golden_qa.yaml`.
2. Add an entry following the existing schema: `id`, `question`, `expected_answer`,
   `expected_documents` (list of PDF filenames), `expected_keywords`.
3. Run `pytest tests/evaluation` to confirm the loader handles the new item.

The `source` field is intentionally absent from individual items (it is a YAML
header comment only). The loader defaults it to `""`.

## Why PDFs are not committed

Source PDFs (CLIP.pdf, SigLIP.pdf, etc.) are research papers under copyright and
would significantly inflate repo size. The golden QA dataset stores only the
filenames as string references. Once PDFs are available locally and indexed into
the VectorStore, replace the mock `retrieved_sources` lists in `test_rag_gates.py`
with real `VectorStore.search()` calls.
