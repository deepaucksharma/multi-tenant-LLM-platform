"""
Document chunking with metadata preservation.
Produces RAG-ready chunks and training-ready segments.
"""
import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List

from loguru import logger

from tenant_data_pipeline.config import TENANTS, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    tenant_id: str
    title: str
    topic: str
    content: str
    chunk_index: int
    total_chunks: int
    char_count: int
    word_count: int
    source_file: str
    metadata: dict = field(default_factory=dict)


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_length: int = MIN_CHUNK_LENGTH,
) -> List[str]:
    """Split text into overlapping chunks, respecting sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{para}".strip() if current_chunk else para
        else:
            if len(current_chunk) >= min_length:
                chunks.append(current_chunk)

            if len(para) > chunk_size:
                sentences = _split_into_sentences(para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= chunk_size:
                        current_chunk = f"{current_chunk} {sent}".strip() if current_chunk else sent
                    else:
                        if len(current_chunk) >= min_length:
                            chunks.append(current_chunk)
                        current_chunk = sent
            else:
                if chunks and overlap > 0:
                    overlap_text = chunks[-1][-overlap:]
                    current_chunk = f"{overlap_text} {para}".strip()
                else:
                    current_chunk = para

    if current_chunk and len(current_chunk) >= min_length:
        chunks.append(current_chunk)

    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_tenant_documents(tenant_id: str) -> List[Chunk]:
    """Chunk all documents for a tenant."""
    config = TENANTS[tenant_id]
    config.chunks_dir.mkdir(parents=True, exist_ok=True)

    redacted_path = config.processed_dir / "redacted_documents.json"
    ingested_path = config.processed_dir / "ingested_documents.json"

    source_path = redacted_path if redacted_path.exists() else ingested_path
    if not source_path.exists():
        logger.warning(f"No documents found for {tenant_id}")
        return []

    with open(source_path) as f:
        documents = json.load(f)

    all_chunks = []

    for doc in documents:
        content = doc.get("content_redacted", doc["content"])
        text_chunks = split_text_into_chunks(content)
        total_chunks = len(text_chunks)

        for idx, chunk_text in enumerate(text_chunks):
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:10]
            chunk = Chunk(
                chunk_id=f"{doc['doc_id']}_chunk_{idx:03d}_{chunk_hash}",
                doc_id=doc["doc_id"],
                tenant_id=tenant_id,
                title=doc["title"],
                topic=doc["topic"],
                content=chunk_text,
                chunk_index=idx,
                total_chunks=total_chunks,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                source_file=doc["source_file"],
                metadata={
                    "file_type": doc.get("file_type", ""),
                    "ingested_at": doc.get("ingested_at", ""),
                },
            )
            all_chunks.append(chunk)

    output_path = config.chunks_dir / "chunks.json"
    output_data = [asdict(c) for c in all_chunks]
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    summary = {
        "tenant_id": tenant_id,
        "total_documents": len(documents),
        "total_chunks": len(all_chunks),
        "avg_chunk_size": sum(c.char_count for c in all_chunks) / max(len(all_chunks), 1),
        "topics_covered": list({c.topic for c in all_chunks}),
        "chunks_per_topic": _chunks_per_topic(all_chunks),
    }
    summary_path = config.chunks_dir / "chunk_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        f"[{tenant_id}] Chunked {len(documents)} docs → {len(all_chunks)} chunks "
        f"(avg {summary['avg_chunk_size']:.0f} chars)"
    )
    return all_chunks


def _chunks_per_topic(chunks: List[Chunk]) -> dict:
    counts: dict = {}
    for c in chunks:
        counts[c.topic] = counts.get(c.topic, 0) + 1
    return counts


def chunk_all_tenants() -> dict:
    results = {}
    for tenant_id in TENANTS:
        chunks = chunk_tenant_documents(tenant_id)
        results[tenant_id] = len(chunks)
    return results


if __name__ == "__main__":
    results = chunk_all_tenants()
    print(f"Chunking results: {results}")
