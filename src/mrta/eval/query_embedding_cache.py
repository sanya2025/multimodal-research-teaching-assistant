"""Frozen query embeddings for the offline benchmark.

The v2 text and caption indices were built with ``nomic-embed-text``, which
:class:`mrta.retrieval.embedder.Embedder` serves through the Ollama REST API.
That makes Ollama a requirement of every *query*, not just of index
construction — so a CI runner with no Ollama cannot evaluate the frozen
benchmark at all, even with the indices checked in.

This module removes that dependency for the frozen queries only. The 100 v2
query vectors are precomputed once against the real embedder and stored
alongside the benchmark; evaluation then reads them instead of calling a
service. Nothing about retrieval changes: the same vectors reach the same
indices in the same order.

The cache is deliberately *not* a general-purpose memoiser. It answers exactly
the query texts it was built for and raises on anything else, because a silent
fallback is the failure mode that matters here — an unnoticed miss would
produce plausible-looking metrics from the wrong vectors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_FORMAT_VERSION = 1


class _EmbedderLike(Protocol):
    """The subset of ``Embedder`` the vector stores actually call."""

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def text_key(text: str) -> str:
    """Stable key for a query string.

    Hashing rather than storing raw text keeps lookup order-independent and the
    npz compact, and sidesteps any encoding ambiguity in the archive.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Content hash of a file, used to bind a cache to one benchmark revision."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class QueryEmbeddingMiss(KeyError):
    """Raised when a query is not present in the frozen cache.

    A distinct type so callers can tell "this benchmark drifted" apart from an
    ordinary lookup failure.
    """


class QueryEmbeddingCache:
    """Query text -> frozen embedding, bound to one model and one query file."""

    def __init__(
        self,
        *,
        keys: list[str],
        vectors: np.ndarray,
        model_name: str,
        queries_sha256: str,
    ) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(keys) != vectors.shape[0]:
            raise ValueError(f"{len(keys)} keys but {vectors.shape[0]} vectors")
        self._by_key = {k: i for k, i in zip(keys, range(len(keys)), strict=True)}
        if len(self._by_key) != len(keys):
            raise ValueError("duplicate query keys in cache")
        self._vectors = vectors.astype("float32", copy=False)
        self._model_name = model_name
        self._queries_sha256 = queries_sha256

    # -- properties ----------------------------------------------------

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def queries_sha256(self) -> str:
        return self._queries_sha256

    def __len__(self) -> int:
        return len(self._by_key)

    # -- lookup --------------------------------------------------------

    def get(self, text: str) -> np.ndarray:
        """Return the frozen vector for ``text``, or raise QueryEmbeddingMiss."""
        key = text_key(text)
        idx = self._by_key.get(key)
        if idx is None:
            raise QueryEmbeddingMiss(
                f"query not in frozen cache (model={self._model_name!r}): {text[:80]!r}"
            )
        return self._vectors[idx]

    def __contains__(self, text: str) -> bool:
        return text_key(text) in self._by_key

    # -- persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ordered by key so the archive bytes depend only on content, never on
        # the order queries happened to be embedded in.
        order = sorted(self._by_key, key=lambda k: k)
        idx = [self._by_key[k] for k in order]
        np.savez_compressed(
            path,
            format_version=np.asarray(CACHE_FORMAT_VERSION),
            keys=np.asarray(order, dtype=object),
            vectors=self._vectors[idx],
            model_name=np.asarray(self._model_name, dtype=object),
            queries_sha256=np.asarray(self._queries_sha256, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> QueryEmbeddingCache:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"query embedding cache not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            version = int(data["format_version"])
            if version != CACHE_FORMAT_VERSION:
                raise ValueError(
                    f"query embedding cache format {version}, expected {CACHE_FORMAT_VERSION}"
                )
            return cls(
                keys=[str(k) for k in data["keys"]],
                vectors=np.asarray(data["vectors"], dtype="float32"),
                model_name=str(data["model_name"]),
                queries_sha256=str(data["queries_sha256"]),
            )

    @classmethod
    def build(
        cls,
        *,
        queries: list[str],
        embedder: _EmbedderLike,
        queries_path: Path,
    ) -> QueryEmbeddingCache:
        """Embed ``queries`` once with the real embedder and freeze the result."""
        unique = list(dict.fromkeys(queries))  # preserve order, drop duplicates
        vectors = embedder.embed(unique)
        return cls(
            keys=[text_key(q) for q in unique],
            vectors=np.asarray(vectors, dtype="float32"),
            model_name=embedder.model_name,
            queries_sha256=file_sha256(queries_path),
        )


class CachedQueryEmbedder:
    """Embedder-shaped facade over a :class:`QueryEmbeddingCache`.

    Drop-in for ``Embedder`` wherever a vector store only ever embeds benchmark
    queries. It has no model, opens no socket and downloads nothing.
    """

    def __init__(self, cache: QueryEmbeddingCache) -> None:
        self._cache = cache

    @property
    def dim(self) -> int:
        return self._cache.dim

    @property
    def model_name(self) -> str:
        return self._cache.model_name

    @property
    def cache(self) -> QueryEmbeddingCache:
        return self._cache

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return frozen vectors for ``texts``; raise on any query not frozen.

        Failing here is the point. Falling back to a live embedder would let a
        drifted benchmark evaluate against vectors from a different model and
        report the difference as a retrieval regression.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.stack([self._cache.get(t) for t in texts]).astype("float32")

    def verify_queries(self, queries_path: Path) -> None:
        """Fail closed if the benchmark file no longer matches the cache."""
        actual = file_sha256(queries_path)
        if actual != self._cache.queries_sha256:
            raise ValueError(
                f"query file {queries_path} has changed since the embedding cache "
                f"was built (cache {self._cache.queries_sha256[:12]}, "
                f"file {actual[:12]}). Rebuild with scripts/build_query_embeddings.py."
            )
