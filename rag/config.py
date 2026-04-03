"""RAG configuration."""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    # Embedding model
    embed_model_name: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_model_path: Optional[str] = None  # Local path override

    # Chroma
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    # Retrieval
    top_k: int = 5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.3

    # Chunking (already done in pipeline, but kept for reference)
    chunk_size: int = 512
    chunk_overlap: int = 64

    # BM25
    use_hybrid: bool = True
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7

    # Reranking
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


RAG_CONFIG = RAGConfig()
