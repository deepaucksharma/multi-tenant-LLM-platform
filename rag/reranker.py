"""
Cross-encoder reranker for improving retrieval precision.
Re-scores retrieved documents against the query using a cross-encoder model.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass

from loguru import logger

from rag.config import RAG_CONFIG

# Lazy-loaded singleton
_reranker_model = None


@dataclass
class RankedResult:
    chunk_id: str
    content: str
    metadata: Dict
    original_score: float
    rerank_score: float
    final_rank: int


def get_reranker():
    """Get or initialize the cross-encoder reranker (singleton)."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker: {RAG_CONFIG.reranker_model}")
            _reranker_model = CrossEncoder(
                RAG_CONFIG.reranker_model,
                max_length=512,
                device="cpu",
            )
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Reranking disabled.")
            _reranker_model = "disabled"
    return _reranker_model


def rerank(
    query: str,
    results: List[Dict],
    top_k: int = 3,
) -> List[RankedResult]:
    """
    Rerank retrieved results using cross-encoder.

    Args:
        query: The user query
        results: List of dicts with 'chunk_id', 'content', 'metadata', 'score'
        top_k: Number of results to return after reranking

    Returns:
        List of RankedResult sorted by rerank score
    """
    if not results:
        return []

    model = get_reranker()

    if model == "disabled" or model is None:
        return [
            RankedResult(
                chunk_id=r.get("chunk_id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                original_score=r.get("score", 0.0),
                rerank_score=r.get("score", 0.0),
                final_rank=i + 1,
            )
            for i, r in enumerate(results[:top_k])
        ]

    pairs = [(query, r["content"]) for r in results]
    scores = model.predict(pairs)

    scored_results = []
    for i, (result, rerank_score) in enumerate(zip(results, scores)):
        scored_results.append({
            "chunk_id": result.get("chunk_id", ""),
            "content": result.get("content", ""),
            "metadata": result.get("metadata", {}),
            "original_score": result.get("score", 0.0),
            "rerank_score": float(rerank_score),
        })

    scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)

    ranked = []
    for i, r in enumerate(scored_results[:top_k]):
        ranked.append(RankedResult(
            chunk_id=r["chunk_id"],
            content=r["content"],
            metadata=r["metadata"],
            original_score=r["original_score"],
            rerank_score=r["rerank_score"],
            final_rank=i + 1,
        ))

    return ranked
