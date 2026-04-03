"""
BM25 keyword index for hybrid retrieval.
Provides sparse keyword matching to complement dense vector search.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from loguru import logger

from tenant_data_pipeline.config import TENANTS


@dataclass
class BM25Result:
    chunk_id: str
    score: float
    content: str
    metadata: Dict


class BM25Index:
    """Simple BM25 implementation for hybrid retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict] = []
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.avg_dl: float = 0.0
        self.n_docs: int = 0
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
            'it', 'its', 'this', 'that', 'these', 'those',
        }
        return [t for t in tokens if len(t) > 1 and t not in stopwords]

    def build(self, documents: List[Dict]):
        """Build BM25 index from document list."""
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_freqs = []

        doc_lengths = []
        df = Counter()

        for doc in documents:
            tokens = self._tokenize(doc["content"])
            doc_lengths.append(len(tokens))
            tf = Counter(tokens)
            self.doc_freqs.append(tf)
            for term in set(tokens):
                df[term] += 1

        self.avg_dl = sum(doc_lengths) / max(self.n_docs, 1)

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        self._built = True
        logger.info(f"BM25 index built: {self.n_docs} documents, {len(self.idf)} unique terms")

    def search(self, query: str, top_k: int = 5) -> List[BM25Result]:
        """Search the BM25 index."""
        if not self._built:
            raise RuntimeError("BM25 index not built. Call build() first.")

        query_tokens = self._tokenize(query)
        scores = []

        for idx, (doc, tf) in enumerate(zip(self.docs, self.doc_freqs)):
            score = 0.0
            doc_len = sum(tf.values())

            for term in query_tokens:
                if term not in self.idf:
                    continue
                term_freq = tf.get(term, 0)
                idf = self.idf[term]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / max(self.avg_dl, 1))
                )
                score += idf * (numerator / max(denominator, 0.001))

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            doc = self.docs[idx]
            results.append(BM25Result(
                chunk_id=doc.get("chunk_id", f"doc_{idx}"),
                score=score,
                content=doc["content"],
                metadata={k: v for k, v in doc.items() if k != "content"},
            ))

        return results


# Cache for tenant BM25 indexes
_bm25_indexes: Dict[str, BM25Index] = {}


def get_bm25_index(tenant_id: str) -> BM25Index:
    """Get or build BM25 index for a tenant."""
    if tenant_id in _bm25_indexes:
        return _bm25_indexes[tenant_id]

    config = TENANTS[tenant_id]
    chunks_path = config.chunks_dir / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks found for tenant {tenant_id}")

    with open(chunks_path) as f:
        chunks = json.load(f)

    index = BM25Index()
    index.build(chunks)
    _bm25_indexes[tenant_id] = index

    return index


def search_bm25(tenant_id: str, query: str, top_k: int = 5) -> List[BM25Result]:
    """Search BM25 index for a tenant."""
    index = get_bm25_index(tenant_id)
    return index.search(query, top_k=top_k)


if __name__ == "__main__":
    for tid in TENANTS:
        try:
            results = search_bm25(tid, "What is the enrollment process?", top_k=3)
            print(f"\n[{tid}] BM25 results:")
            for r in results:
                print(f"  Score: {r.score:.3f} | {r.content[:80]}...")
        except FileNotFoundError as e:
            print(f"[{tid}] {e}")
