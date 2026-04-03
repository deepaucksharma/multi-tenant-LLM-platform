"""
Build tenant-isolated vector indexes in ChromaDB.
Each tenant gets its own Chroma collection with metadata.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from loguru import logger

from rag.config import RAG_CONFIG
from rag.embeddings import embed_texts, get_embedding_dimension
from tenant_data_pipeline.config import TENANTS


def get_chroma_client() -> chromadb.ClientAPI:
    """Get persistent Chroma client."""
    persist_dir = RAG_CONFIG.chroma_persist_dir
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def get_collection_name(tenant_id: str) -> str:
    """Generate collection name for a tenant."""
    return f"tenant_{tenant_id}_docs"


def build_tenant_index(tenant_id: str, force_rebuild: bool = False) -> Dict:
    """Build vector index for a single tenant."""
    config = TENANTS[tenant_id]
    chunks_path = config.chunks_dir / "chunks.json"

    if not chunks_path.exists():
        logger.error(f"[{tenant_id}] No chunks found at {chunks_path}. Run data pipeline first.")
        return {"tenant_id": tenant_id, "status": "error", "reason": "no chunks"}

    with open(chunks_path) as f:
        chunks = json.load(f)

    if not chunks:
        logger.warning(f"[{tenant_id}] Empty chunks file")
        return {"tenant_id": tenant_id, "status": "error", "reason": "empty chunks"}

    client = get_chroma_client()
    collection_name = get_collection_name(tenant_id)

    # Delete existing collection if force rebuild
    if force_rebuild:
        try:
            client.delete_collection(collection_name)
            logger.info(f"[{tenant_id}] Deleted existing collection: {collection_name}")
        except Exception:
            pass

    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "tenant_id": tenant_id,
            "domain": config.domain,
            "hnsw:space": "cosine",
        },
    )

    # Check if already populated
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info(
            f"[{tenant_id}] Collection already has {existing_count} documents. "
            f"Use force_rebuild=True to rebuild."
        )
        return {
            "tenant_id": tenant_id,
            "status": "exists",
            "document_count": existing_count,
        }

    # Prepare data for indexing
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "tenant_id": chunk["tenant_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "topic": chunk["topic"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "source_file": chunk["source_file"],
            "char_count": chunk["char_count"],
            "word_count": chunk["word_count"],
        }
        for chunk in chunks
    ]

    # Generate embeddings
    logger.info(f"[{tenant_id}] Embedding {len(texts)} chunks...")
    t0 = time.time()
    embeddings = embed_texts(texts, show_progress=True)
    embed_time = round(time.time() - t0, 2)
    logger.info(f"[{tenant_id}] Embedding complete in {embed_time}s")

    # Index in batches (Chroma has limits on batch size)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )

    final_count = collection.count()
    logger.info(f"[{tenant_id}] Indexed {final_count} chunks into collection '{collection_name}'")

    return {
        "tenant_id": tenant_id,
        "status": "success",
        "collection_name": collection_name,
        "document_count": final_count,
        "embedding_time_sec": embed_time,
        "embedding_dimension": embeddings.shape[1],
    }


def build_all_indexes(force_rebuild: bool = False) -> Dict:
    """Build indexes for all tenants."""
    results = {}
    for tenant_id in TENANTS:
        results[tenant_id] = build_tenant_index(tenant_id, force_rebuild=force_rebuild)
    return results


def list_collections() -> List[Dict]:
    """List all existing collections."""
    client = get_chroma_client()
    collections = client.list_collections()
    info = []
    for col in collections:
        c = client.get_collection(col.name)
        info.append({
            "name": col.name,
            "count": c.count(),
            "metadata": col.metadata,
        })
    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RAG indexes")
    parser.add_argument("--force", action="store_true", help="Force rebuild indexes")
    parser.add_argument("--list", action="store_true", help="List existing collections")
    args = parser.parse_args()

    if args.list:
        collections = list_collections()
        print("\nExisting collections:")
        for c in collections:
            print(f"  {c['name']}: {c['count']} documents")
    else:
        results = build_all_indexes(force_rebuild=args.force)
        print("\nIndex build results:")
        for tid, result in results.items():
            print(f"  [{tid}] {result['status']}: {result.get('document_count', 0)} documents")
