"""mrta.core.rag_pipeline — retrieve → [rerank] → prompt → generate."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mrta.core.schemas import Chunk

if TYPE_CHECKING:
    from mrta.core.llm import LLMClient
    from mrta.retrieval.reranker import Reranker
    from mrta.retrieval.vector_store import VectorStore

SYSTEM = "You answer questions grounded in the provided context. Always cite pages."


def rag_query(
    question: str,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = 5,
    reranker: Reranker | None = None,
    rerank_top_n: int = 3,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    When reranker is provided, vector-search candidates are re-scored by the
    cross-encoder and only the top rerank_top_n chunks are passed to the LLM.

    Returns:
        {"answer": str, "sources": list[Chunk], "scores": list[float], "latency_s": float}
    """
    from mrta.observability.tracing import trace_span
    from mrta.prompts import load_prompt

    t0 = time.time()
    with trace_span("mrta.rag_query") as span:
        span.set_attribute("query.length", len(question))
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("reranker.enabled", reranker is not None)
        span.set_attribute("reranker.top_n", rerank_top_n)
        span.set_attribute("model.llm", llm.model)
        retrieved = vector_store.search_with_scores(question, k=top_k)
        sources: list[Chunk] = [c for c, _ in retrieved]
        retrieval_scores: list[float] = [s for _, s in retrieved]
        span.set_attribute("retrieval.chunk_count", len(sources))
        if reranker is not None:
            sources = reranker.rerank(question, sources, top_n=rerank_top_n)
            retrieval_scores = retrieval_scores[: len(sources)]
        prompt = load_prompt("rag", chunks=sources, question=question)
        answer: str = llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        )
        latency_s = time.time() - t0
        span.set_attribute("latency_ms", latency_s * 1000)
    return {"answer": answer, "sources": sources, "scores": retrieval_scores, "latency_s": latency_s}
