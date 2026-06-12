"""mrta.evaluation.eval_pipeline — run all metrics over a benchmark."""

from __future__ import annotations

from mrta.core.llm import LLMClient
from mrta.core.rag_pipeline import rag_query
from mrta.core.schemas import EvalReport
from mrta.evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    faithfulness,
    hallucination_rate,
)
from mrta.retrieval.vector_store import VectorStore


def run_eval(
    benchmark: list[dict],
    vs: VectorStore,
    llm: LLMClient,
    top_k: int = 5,
) -> EvalReport:
    """Run all four metrics over benchmark and return an averaged EvalReport.

    Each benchmark item must have a "question" key.
    """
    from mrta.observability.tracing import trace_span

    ar_scores: list[float] = []
    f_scores: list[float] = []
    cc_scores: list[float] = []
    hr_scores: list[float] = []
    latencies: list[float] = []

    with trace_span("mrta.evaluation", {"benchmark.size": len(benchmark)}):
        for item in benchmark:
            result = rag_query(item["question"], vs, llm, top_k=top_k)
            ar_scores.append(answer_relevance(item["question"], result["answer"]))
            f_scores.append(faithfulness(result["answer"], result["sources"]))
            cc_scores.append(citation_correctness(result["answer"], result["sources"]))
            hr_scores.append(hallucination_rate(result["answer"], result["sources"]))
            latencies.append(result["latency_s"])

        n = len(benchmark)
        return EvalReport(
            n_questions=n,
            answer_relevance=sum(ar_scores) / n,
            faithfulness=sum(f_scores) / n,
            citation_correctness=sum(cc_scores) / n,
            hallucination_rate=sum(hr_scores) / n,
            mean_latency_s=sum(latencies) / n,
        )


__all__ = ["run_eval"]
