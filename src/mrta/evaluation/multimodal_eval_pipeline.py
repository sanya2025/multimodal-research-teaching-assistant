"""mrta.evaluation.multimodal_eval_pipeline — evaluation loop for MultimodalRAG.

Drives MultimodalRAG.ask() over a benchmark dataset and aggregates
figure_recall_at_k, multimodal_recall_at_k, and multimodal_citation_correctness
alongside the existing text-only metrics (answer_relevance, faithfulness).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mrta.evaluation.metrics import answer_relevance, faithfulness
from mrta.evaluation.multimodal_metrics import (
    figure_recall_at_k,
    multimodal_citation_correctness,
    multimodal_recall_at_k,
)


@dataclass
class MultimodalEvalReport:
    """Aggregated evaluation results over a multimodal benchmark."""

    n_questions: int

    # retrieval
    figure_recall_at_1: float = 0.0
    figure_recall_at_3: float = 0.0
    figure_recall_at_5: float = 0.0
    text_recall_at_k: float = 0.0
    visual_recall_at_k: float = 0.0
    overall_recall_at_k: float = 0.0

    # generation
    answer_relevance: float = 0.0
    faithfulness: float = 0.0

    # citation
    citation_format: float = 0.0
    citation_provenance: float = 0.0
    citation_support: float = 0.0
    citation_overall: float = 0.0

    # latency
    mean_latency_s: float = 0.0

    # per-question-type breakdowns (populated when question_type is present)
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)


def run_multimodal_eval(
    benchmark: list[dict],
    rag,  # MultimodalRAG — typed as Any to avoid circular import in tests
    eval_k: int = 5,
) -> MultimodalEvalReport:
    """Run all multimodal metrics over benchmark and return an averaged report.

    Each benchmark item must have a ``question`` key. Optional keys:
        ``relevant_text_pages`` (list[int])
        ``relevant_figures`` (list[dict])
        ``question_type`` (str)

    Faithfulness is computed against the text evidence retrieved by the RAG
    pipeline — the same limitation as in text-only eval.

    A text-only judge cannot independently verify raw visual semantics.
    When the answer cites visual evidence that has no textual representation,
    faithfulness scores will be pessimistically low. That limitation is
    intentional and is documented in the returned report via ``citation_support``.
    """
    from mrta.core.schemas import Chunk  # local import avoids heavy top-level dep

    fr1_scores: list[float] = []
    fr3_scores: list[float] = []
    fr5_scores: list[float] = []
    text_r_scores: list[float] = []
    visual_r_scores: list[float] = []
    overall_r_scores: list[float] = []
    ar_scores: list[float] = []
    faith_scores: list[float] = []
    cit_fmt: list[float] = []
    cit_prov: list[float] = []
    cit_sup: list[float] = []
    cit_all: list[float] = []
    latencies: list[float] = []

    by_type: dict[str, list[dict[str, float]]] = {}

    for item in benchmark:
        question = item["question"]
        expected_text_pages: list[int] = item.get("relevant_text_pages", [])
        expected_figures: list[dict] = item.get("relevant_figures", [])
        qtype: str = item.get("question_type", "unknown")

        answer_obj = rag.ask(question)

        # reconstruct retrieved evidence from citations for metric computation
        all_citations = answer_obj.text_citations + answer_obj.visual_citations

        # figure recall at 1, 3, 5
        # retrieved is approximated from text_citations for page matching
        text_chunks = [
            Chunk(
                chunk_id=c.evidence_id,
                doc_id="eval",
                source=c.source,
                page=c.page,
                text=answer_obj.answer[:50],
            )
            for c in answer_obj.text_citations
        ]

        # build a flat EvidenceRecord list from citations for recall metrics
        from mrta.core.schemas import EvidenceRecord

        evidence_records = [
            EvidenceRecord(
                evidence_id=c.evidence_id,
                doc_id="eval",
                source=c.source,
                page=c.page,
                modality=c.modality,
                figure_index=c.figure_index,
            )
            for c in all_citations
        ]

        fr1_scores.append(figure_recall_at_k(evidence_records, expected_figures, k=1))
        fr3_scores.append(figure_recall_at_k(evidence_records, expected_figures, k=3))
        fr5_scores.append(figure_recall_at_k(evidence_records, expected_figures, k=5))

        mm_r = multimodal_recall_at_k(
            evidence_records, expected_text_pages, expected_figures, k=eval_k
        )
        text_r_scores.append(mm_r["text"])
        visual_r_scores.append(mm_r["visual"])
        overall_r_scores.append(mm_r["overall"])

        ar_scores.append(answer_relevance(question, answer_obj.answer))
        faith_scores.append(faithfulness(answer_obj.answer, text_chunks))

        cit = multimodal_citation_correctness(answer_obj, evidence_records)
        cit_fmt.append(cit["format"])
        cit_prov.append(cit["provenance"])
        cit_sup.append(cit["support"])
        cit_all.append(cit["overall"])

        latencies.append(answer_obj.latency_s)

        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(
            {
                "figure_recall_at_5": fr5_scores[-1],
                "overall_recall": mm_r["overall"],
                "answer_relevance": ar_scores[-1],
                "citation_overall": cit_all[-1],
            }
        )

    n = len(benchmark)

    def _avg(lst: list[float]) -> float:
        return sum(lst) / n if n else 0.0

    by_type_agg: dict[str, dict[str, float]] = {
        qtype: {metric: sum(v[metric] for v in rows) / len(rows) for metric in rows[0]}
        for qtype, rows in by_type.items()
        if rows
    }

    return MultimodalEvalReport(
        n_questions=n,
        figure_recall_at_1=_avg(fr1_scores),
        figure_recall_at_3=_avg(fr3_scores),
        figure_recall_at_5=_avg(fr5_scores),
        text_recall_at_k=_avg(text_r_scores),
        visual_recall_at_k=_avg(visual_r_scores),
        overall_recall_at_k=_avg(overall_r_scores),
        answer_relevance=_avg(ar_scores),
        faithfulness=_avg(faith_scores),
        citation_format=_avg(cit_fmt),
        citation_provenance=_avg(cit_prov),
        citation_support=_avg(cit_sup),
        citation_overall=_avg(cit_all),
        mean_latency_s=_avg(latencies),
        by_type=by_type_agg,
    )


__all__ = ["MultimodalEvalReport", "run_multimodal_eval"]
