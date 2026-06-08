"""mrta.evaluation — evaluation metrics and pipeline."""

from mrta.evaluation.eval_pipeline import run_eval
from mrta.evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    faithfulness,
    hallucination_rate,
)

__all__ = [
    "run_eval",
    "answer_relevance",
    "faithfulness",
    "citation_correctness",
    "hallucination_rate",
]
