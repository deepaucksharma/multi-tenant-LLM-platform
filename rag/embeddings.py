"""
Embedding model management.
Uses sentence-transformers for local embedding generation.
Supports batched encoding for efficiency.
"""
import os
import hashlib
from typing import List

import numpy as np
from loguru import logger

from rag.config import RAG_CONFIG

# Lazy-loaded singleton
_embed_model = None


class _FallbackEmbedModel:
    """
    Deterministic hash-based embedding fallback for offline environments.

    This is not semantically strong, but it keeps retrieval, indexing, and
    evaluation code paths operational when no sentence-transformer weights are
    available locally and the runtime cannot reach the network.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **_: object,
    ):
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else list(texts)
        vectors = np.vstack([self._embed_text(text) for text in text_list])
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        if single_input:
            return vectors[0] if convert_to_numpy else vectors[0].tolist()
        return vectors if convert_to_numpy else vectors.tolist()

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec


def get_embed_model():
    """Get or initialize the embedding model (singleton)."""
    global _embed_model
    if _embed_model is None:
        model_path = RAG_CONFIG.embed_model_path or RAG_CONFIG.embed_model_name
        logger.info(f"Loading embedding model: {model_path}")
        try:
            from sentence_transformers import SentenceTransformer

            local_only = os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
            _embed_model = SentenceTransformer(
                model_path,
                device="cpu",  # Embeddings on CPU to save GPU for LLM
                local_files_only=local_only,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falling back to deterministic local embeddings because the configured "
                f"embedding model could not be loaded: {exc}"
            )
            _embed_model = _FallbackEmbedModel()

        logger.info(
            f"Embedding model loaded. Dimension: {_embed_model.get_sentence_embedding_dimension()}"
        )
    return _embed_model


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed a list of texts, returning numpy array of shape (N, dim)."""
    model = get_embed_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # For cosine similarity
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> List[float]:
    """Embed a single query string, returning a list of floats."""
    model = get_embed_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()


def get_embedding_dimension() -> int:
    """Return the embedding vector dimension."""
    model = get_embed_model()
    return model.get_sentence_embedding_dimension()
