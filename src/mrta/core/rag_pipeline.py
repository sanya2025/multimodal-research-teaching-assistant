"""mrta.core.rag_pipeline — retrieve → prompt → generate."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mrta.core.schemas import Chunk

if TYPE_CHECKING:
    from mrta.core.llm import LLMClient
    from mrta.retrieval.vector_store import VectorStore

SYSTEM = "You answer questions grounded in the provided context. Always cite pages."


def rag_query(
    question: str,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = 5,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    Returns:
        {"answer": str, "sources": list[Chunk], "latency_s": float}
    """
    from mrta.prompts import load_prompt

    t0 = time.time()
    sources: list[Chunk] = vector_store.search(question, k=top_k)
    prompt = load_prompt("rag", chunks=sources, question=question)
    answer: str = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    )
    latency_s = time.time() - t0
    return {"answer": answer, "sources": sources, "latency_s": latency_s}
