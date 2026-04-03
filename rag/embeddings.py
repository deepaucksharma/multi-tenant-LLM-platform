"""
Embedding model management.
Uses sentence-transformers for local embedding generation.
Supports batched encoding for efficiency.
"""
import os
from typing import List, Optional
from pathlib import Path

import numpy as np
from loguru import logger

from rag.config import RAG_CONFIG

# Lazy-loaded singleton
_embed_model = None


def get_embed_model():
    """Get or initialize the embedding model (singleton)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        model_path = RAG_CONFIG.embed_model_path or RAG_CONFIG.embed_model_name
        logger.info(f"Loading embedding model: {model_path}")

        _embed_model = SentenceTransformer(
            model_path,
            device="cpu",  # Embeddings on CPU to save GPU for LLM
        )
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
