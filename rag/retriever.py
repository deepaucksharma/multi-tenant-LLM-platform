"""
Multi-tenant retriever with hybrid search (semantic + BM25) and reranking.
This is the main retrieval interface used by the inference layer.
"""
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

import chromadb
from loguru import logger

from rag.config import RAG_CONFIG
from rag.embeddings import embed_query
from rag.build_index import get_chroma_client, get_collection_name
from rag.bm25_index import search_bm25, BM25Result
from rag.reranker import rerank, RankedResult

# Imported lazily to avoid circular imports at module load time
# Used for defense-in-depth tenant isolation check after retrieval
def _validate_isolation(tenant_id: str, chunks: List) -> List:
    """Filter out any cross-tenant chunks as a defense-in-depth layer."""
    try:
        from inference.tenant_router import validate_tenant_isolation
        return validate_tenant_isolation(tenant_id, [
            {"tenant_id": c.tenant_id, "chunk_id": c.chunk_id, "metadata": {}}
            for c in chunks
        ])
    except Exception:
        # If the router is not available (e.g., during data-only runs), skip
        return [{"tenant_id": c.tenant_id, "chunk_id": c.chunk_id} for c in chunks]


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with all metadata."""
    chunk_id: str
    content: str
    title: str
    topic: str
    source_file: str
    tenant_id: str
    score: float
    retrieval_method: str  # "semantic", "bm25", "hybrid"
    rank: int
    citation_key: str = ""

    def __post_init__(self):
        if not self.citation_key:
            self.citation_key = f"[{self.title} — {self.topic}]"


@dataclass
class RetrievalResult:
    """Complete retrieval result for a query."""
    query: str
    tenant_id: str
    chunks: List[RetrievedChunk]
    total_retrieved: int
    retrieval_method: str
    retrieval_time_ms: float
    metadata: Dict = field(default_factory=dict)


class TenantRetriever:
    """Retriever that enforces tenant isolation and supports hybrid search."""

    def __init__(self):
        self._chroma_client: Optional[chromadb.ClientAPI] = None

    @property
    def chroma_client(self) -> chromadb.ClientAPI:
        if self._chroma_client is None:
            self._chroma_client = get_chroma_client()
        return self._chroma_client

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = None,
        use_hybrid: bool = None,
        use_reranker: bool = None,
        topic_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query within a tenant's scope.

        Args:
            query: User query
            tenant_id: Tenant identifier (enforces isolation)
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid retrieval
            use_reranker: Whether to use cross-encoder reranking
            topic_filter: Optional topic filter

        Returns:
            RetrievalResult with ranked chunks
        """
        top_k = top_k or RAG_CONFIG.rerank_top_k
        use_hybrid = use_hybrid if use_hybrid is not None else RAG_CONFIG.use_hybrid
        use_reranker = use_reranker if use_reranker is not None else RAG_CONFIG.use_reranker

        t0 = time.time()

        candidate_k = max(top_k * 3, RAG_CONFIG.top_k)

        # --- Semantic search ---
        semantic_results = self._semantic_search(
            query, tenant_id, candidate_k, topic_filter
        )

        # --- BM25 search ---
        bm25_results = []
        if use_hybrid:
            try:
                bm25_results = self._bm25_search(query, tenant_id, candidate_k)
            except Exception as e:
                logger.warning(f"BM25 search failed for {tenant_id}: {e}")

        # --- Merge results ---
        if use_hybrid and bm25_results:
            merged = self._merge_hybrid(
                semantic_results, bm25_results,
                semantic_weight=RAG_CONFIG.semantic_weight,
                bm25_weight=RAG_CONFIG.bm25_weight,
            )
            retrieval_method = "hybrid"
        else:
            merged = semantic_results
            retrieval_method = "semantic"

        # --- Rerank ---
        if use_reranker and len(merged) > top_k:
            reranked = rerank(query, merged, top_k=top_k)
            final_chunks = [
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    title=r.metadata.get("title", ""),
                    topic=r.metadata.get("topic", ""),
                    source_file=r.metadata.get("source_file", ""),
                    tenant_id=tenant_id,
                    score=r.rerank_score,
                    retrieval_method=f"{retrieval_method}+reranked",
                    rank=r.final_rank,
                )
                for r in reranked
            ]
        else:
            final_chunks = [
                RetrievedChunk(
                    chunk_id=r.get("chunk_id", ""),
                    content=r.get("content", ""),
                    title=r.get("metadata", {}).get("title", ""),
                    topic=r.get("metadata", {}).get("topic", ""),
                    source_file=r.get("metadata", {}).get("source_file", ""),
                    tenant_id=tenant_id,
                    score=r.get("score", 0.0),
                    retrieval_method=retrieval_method,
                    rank=i + 1,
                )
                for i, r in enumerate(merged[:top_k])
            ]

        retrieval_time = round((time.time() - t0) * 1000, 2)

        # Defense-in-depth: strip any chunks that don't belong to this tenant
        # before they can contaminate the prompt context.
        clean_ids = {c["chunk_id"] for c in _validate_isolation(tenant_id, final_chunks)}
        final_chunks = [c for c in final_chunks if c.chunk_id in clean_ids]

        return RetrievalResult(
            query=query,
            tenant_id=tenant_id,
            chunks=final_chunks,
            total_retrieved=len(final_chunks),
            retrieval_method=retrieval_method + ("+reranked" if use_reranker else ""),
            retrieval_time_ms=retrieval_time,
            metadata={
                "semantic_candidates": len(semantic_results),
                "bm25_candidates": len(bm25_results),
                "topic_filter": topic_filter,
            },
        )

    def _semantic_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        topic_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Perform semantic search using ChromaDB."""
        collection_name = get_collection_name(tenant_id)

        try:
            collection = self.chroma_client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Collection not found for tenant {tenant_id}: {e}")
            return []

        query_embedding = embed_query(query)

        try:
            where_clause = {"tenant_id": tenant_id}
            if topic_filter:
                where_clause = {
                    "$and": [
                        {"tenant_id": tenant_id},
                        {"topic": topic_filter},
                    ]
                }
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # SECURITY: Do NOT fall back to an unfiltered query — that would
            # expose all tenants' data. Return empty instead.
            logger.error(
                f"Chroma filtered query failed for tenant '{tenant_id}' "
                f"(NOT retrying without filter for security): {e}"
            )
            return []

        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance

                if similarity < RAG_CONFIG.similarity_threshold:
                    continue

                formatted.append({
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": similarity,
                })

        return formatted

    def _bm25_search(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """Perform BM25 keyword search."""
        bm25_results = search_bm25(tenant_id, query, top_k=top_k)

        if bm25_results:
            max_score = max(r.score for r in bm25_results)
            min_score = min(r.score for r in bm25_results)
            score_range = max_score - min_score if max_score != min_score else 1.0
        else:
            return []

        formatted = []
        for r in bm25_results:
            normalized_score = (r.score - min_score) / score_range if score_range > 0 else 0.5
            formatted.append({
                "chunk_id": r.chunk_id,
                "content": r.content,
                "metadata": r.metadata,
                "score": normalized_score,
            })

        return formatted

    def _merge_hybrid(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[Dict]:
        """Merge semantic and BM25 results using weighted reciprocal rank fusion."""
        scores: Dict[str, Dict] = {}

        for rank, result in enumerate(semantic_results):
            cid = result["chunk_id"]
            rrf_score = 1.0 / (rank + 60)  # RRF constant = 60
            scores[cid] = {
                "chunk_id": cid,
                "content": result["content"],
                "metadata": result["metadata"],
                "semantic_score": result["score"],
                "bm25_score": 0.0,
                "rrf_semantic": rrf_score * semantic_weight,
                "rrf_bm25": 0.0,
            }

        for rank, result in enumerate(bm25_results):
            cid = result["chunk_id"]
            rrf_score = 1.0 / (rank + 60)
            if cid in scores:
                scores[cid]["bm25_score"] = result["score"]
                scores[cid]["rrf_bm25"] = rrf_score * bm25_weight
            else:
                scores[cid] = {
                    "chunk_id": cid,
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "bm25_score": result["score"],
                    "rrf_semantic": 0.0,
                    "rrf_bm25": rrf_score * bm25_weight,
                }

        for cid in scores:
            scores[cid]["score"] = scores[cid]["rrf_semantic"] + scores[cid]["rrf_bm25"]

        merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        return merged


# Module-level singleton
_retriever: Optional[TenantRetriever] = None


def get_retriever() -> TenantRetriever:
    """Get or create the singleton retriever."""
    global _retriever
    if _retriever is None:
        _retriever = TenantRetriever()
    return _retriever


def retrieve(
    query: str,
    tenant_id: str,
    top_k: int = 3,
    **kwargs,
) -> RetrievalResult:
    """Convenience function for retrieval."""
    retriever = get_retriever()
    return retriever.retrieve(query, tenant_id, top_k=top_k, **kwargs)


def format_context_for_llm(result: RetrievalResult) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    if not result.chunks:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for chunk in result.chunks:
        citation = chunk.citation_key
        context_parts.append(
            f"--- Source: {citation} (relevance: {chunk.score:.2f}) ---\n"
            f"{chunk.content}\n"
        )

    return "\n".join(context_parts)


def format_citations(result: RetrievalResult) -> List[Dict]:
    """Extract citation information from retrieval results."""
    citations = []
    for chunk in result.chunks:
        citations.append({
            "citation_key": chunk.citation_key,
            "title": chunk.title,
            "topic": chunk.topic,
            "source_file": chunk.source_file,
            "relevance_score": round(chunk.score, 3),
            "retrieval_method": chunk.retrieval_method,
        })
    return citations


if __name__ == "__main__":
    test_queries = {
        "sis": [
            "What is the enrollment process for new students?",
            "How does FERPA protect student records?",
            "What happens after 5 unexcused absences?",
        ],
        "mfg": [
            "What is the assembly line startup procedure?",
            "How are critical defects handled?",
            "What is the lockout tagout procedure?",
        ],
    }

    for tenant_id, queries in test_queries.items():
        print(f"\n{'='*70}")
        print(f"TENANT: {tenant_id.upper()}")
        print(f"{'='*70}")

        for query in queries:
            result = retrieve(query, tenant_id, top_k=3)
            print(f"\nQuery: {query}")
            print(f"Method: {result.retrieval_method} | Time: {result.retrieval_time_ms}ms")
            for chunk in result.chunks:
                print(f"  [{chunk.rank}] Score: {chunk.score:.3f} | {chunk.citation_key}")
                print(f"      {chunk.content[:100]}...")

    # Cross-tenant isolation test
    print(f"\n{'='*70}")
    print("CROSS-TENANT ISOLATION TEST")
    print(f"{'='*70}")
    result = retrieve("What is the FERPA compliance process?", "mfg", top_k=3)
    print(f"\nAsking SIS question to MFG tenant:")
    print(f"Results: {result.total_retrieved} chunks")
    for chunk in result.chunks:
        print(f"  Score: {chunk.score:.3f} | Topic: {chunk.topic}")
    if result.total_retrieved == 0 or all(c.score < 0.5 for c in result.chunks):
        print("  No relevant cross-tenant leakage")
    else:
        print("  Potential cross-tenant content found")
