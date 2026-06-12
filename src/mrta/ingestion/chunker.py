"""Chunking strategies — converts a PdfDocument into retrievable Chunk objects."""

from __future__ import annotations

import re

from mrta.core.exceptions import IngestionError
from mrta.core.schemas import Chunk, PdfDocument


def fixed_chunks(pdf: PdfDocument, size: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Split each page into fixed-size character windows with overlap."""
    out = []
    for page in pdf.pages:
        text = page.text
        i, idx = 0, 0
        while i < len(text):
            piece = text[i : i + size]
            if piece.strip():
                out.append(
                    Chunk(
                        chunk_id=f"{page.doc_id}_p{page.page}_c{idx}",
                        doc_id=page.doc_id,
                        source=page.source,
                        page=page.page,
                        text=piece,
                    )
                )
                idx += 1
            i += size - overlap
    return out


def recursive_chunks(pdf: PdfDocument, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split using RecursiveCharacterTextSplitter — respects paragraph/sentence boundaries."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    out = []
    for page in pdf.pages:
        for idx, piece in enumerate(splitter.split_text(page.text)):
            if piece.strip():
                out.append(
                    Chunk(
                        chunk_id=f"{page.doc_id}_p{page.page}_c{idx}",
                        doc_id=page.doc_id,
                        source=page.source,
                        page=page.page,
                        text=piece,
                    )
                )
    return out


def token_chunks(pdf: PdfDocument, size: int = 700, overlap: int = 100) -> list[Chunk]:
    """Split by token count using tiktoken cl100k_base encoding."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    out = []
    for page in pdf.pages:
        tokens = enc.encode(page.text)
        i, idx = 0, 0
        while i < len(tokens):
            window = tokens[i : i + size]
            piece = enc.decode(window).strip()
            if piece:
                out.append(
                    Chunk(
                        chunk_id=f"{page.doc_id}_p{page.page}_c{idx}",
                        doc_id=page.doc_id,
                        source=page.source,
                        page=page.page,
                        text=piece,
                        n_tokens=len(window),
                    )
                )
                idx += 1
            i += size - overlap
    return out


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def semantic_chunks(
    pdf: PdfDocument,
    similarity_threshold: float = 0.55,
    max_chars: int = 1200,
) -> list[Chunk]:
    """Merge adjacent sentences while embedding cosine similarity stays above threshold."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    out = []
    for page in pdf.pages:
        sentences = _split_sentences(page.text)
        if not sentences:
            continue
        embs = embedder.encode(sentences, normalize_embeddings=True)
        current, current_emb = [sentences[0]], embs[0]
        chunk_idx = 0
        for sent, emb in zip(sentences[1:], embs[1:]):
            sim = float(np.dot(current_emb, emb))
            combined = " ".join(current + [sent])
            if sim >= similarity_threshold and len(combined) <= max_chars:
                current.append(sent)
                current_emb = (current_emb + emb) / 2
            else:
                out.append(
                    Chunk(
                        chunk_id=f"{page.doc_id}_p{page.page}_c{chunk_idx}",
                        doc_id=page.doc_id,
                        source=page.source,
                        page=page.page,
                        text=" ".join(current),
                    )
                )
                chunk_idx += 1
                current, current_emb = [sent], emb
        if current:
            out.append(
                Chunk(
                    chunk_id=f"{page.doc_id}_p{page.page}_c{chunk_idx}",
                    doc_id=page.doc_id,
                    source=page.source,
                    page=page.page,
                    text=" ".join(current),
                )
            )
    return out


_STRATEGIES: dict[str, object] = {
    "fixed": fixed_chunks,
    "recursive": recursive_chunks,
    "token": token_chunks,
    "semantic": semantic_chunks,
}


def chunk_pdf(
    pdf: PdfDocument,
    strategy: str = "recursive",
    **kwargs: object,
) -> list[Chunk]:
    """Dispatch to a chunking strategy by name.

    Strategies: 'fixed', 'recursive' (default), 'token', 'semantic'.
    """
    from mrta.observability.tracing import trace_span

    if strategy not in _STRATEGIES:
        raise IngestionError(f"Unknown strategy {strategy!r}; choose from {list(_STRATEGIES)}")
    fn = _STRATEGIES[strategy]
    doc_source = pdf.pages[0].source if pdf.pages else ""
    with trace_span(
        "mrta.ingestion",
        {"document.path": doc_source, "chunk.strategy": strategy},
    ) as span:
        result: list[Chunk] = fn(pdf, **kwargs)  # type: ignore[operator]
        span.set_attribute("chunk.count", len(result))
    return result
