"""Evaluation configuration and shared utilities."""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from loguru import logger


@dataclass
class EvalResult:
    """Single evaluation result."""
    test_id: str
    tenant_id: str
    category: str
    question: str
    expected_answer: str
    model_answer: str
    scores: Dict[str, float] = field(default_factory=dict)
    passed: bool = True
    flags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """Complete evaluation report."""
    report_id: str
    tenant_id: str
    model_version: str
    eval_type: str
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    scores: Dict[str, float]
    results: List[Dict]
    summary: str

    def to_dict(self):
        return asdict(self)

    def save(self, directory: str = "evaluation/reports"):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{self.eval_type}_{self.tenant_id}_{self.report_id}.json"
        filepath.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        logger.info(f"Report saved: {filepath}")
        return filepath


def load_golden_set(tenant_id: str) -> List[Dict]:
    """Load golden test set for a tenant."""
    path = Path(f"evaluation/golden_sets/{tenant_id}_golden.json")
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    with open(path) as f:
        return json.load(f)


def compute_keyword_overlap(response: str, required_elements: List[str]) -> float:
    """Check what fraction of required elements appear in the response."""
    if not required_elements:
        return 1.0
    response_lower = response.lower()
    found = sum(1 for elem in required_elements if elem.lower() in response_lower)
    return found / len(required_elements)


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two texts using embeddings."""
    try:
        from rag.embeddings import get_embed_model
        import numpy as np
        model = get_embed_model()
        emb_a = model.encode(text_a, normalize_embeddings=True)
        emb_b = model.encode(text_b, normalize_embeddings=True)
        similarity = float(np.dot(emb_a, emb_b))
        return max(0.0, min(1.0, similarity))
    except Exception:
        # Fallback: simple word overlap
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        return overlap / max(len(words_a), len(words_b))
