"""mrta.retrieval.embedder — thin wrapper around sentence-transformers or Ollama."""

from __future__ import annotations

import numpy as np


class Embedder:
    """Encodes text to L2-normalized float32 vectors.

    Uses sentence-transformers for HuggingFace model IDs (contain '/').
    Uses the Ollama REST API for other model names (e.g. 'nomic-embed-text').
    Model selection reads settings.embedding_model by default; model_name overrides it.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from mrta.core.config import settings

        self._model_name: str = model_name or settings.embedding_model
        self._st_model = None
        self._dim_cache: int | None = None
        # "/" in name → HuggingFace / sentence-transformers path; otherwise → Ollama
        self._use_st: bool = "/" in self._model_name

    @property
    def model_name(self) -> str:
        """The model identifier used by this embedder."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Embedding dimensionality (lazy — loads model on first call)."""
        if self._dim_cache is not None:
            return self._dim_cache
        if self._use_st:
            self._dim_cache = self._ensure_st().get_embedding_dimension()
        else:
            # Probe with a single embed to discover dim
            self._dim_cache = self._embed_ollama(["probe"]).shape[1]
        return self._dim_cache

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim), L2-normalized."""
        if self._use_st:
            return (
                self._ensure_st()
                .encode(
                    texts,
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                .astype("float32")
            )
        return self._embed_ollama(texts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_st(self) -> object:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self._model_name)
        return self._st_model

    def _embed_ollama(self, texts: list[str]) -> np.ndarray:
        import httpx

        from mrta.core.config import settings

        vecs: list[list[float]] = []
        for text in texts:
            resp = httpx.post(
                f"{settings.ollama_host}/api/embeddings",
                json={"model": self._model_name, "prompt": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            vecs.append(resp.json()["embedding"])

        arr = np.array(vecs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)
