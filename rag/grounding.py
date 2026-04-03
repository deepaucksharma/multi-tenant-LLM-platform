"""
Grounding verification for RAG responses.
Checks if LLM output is supported by retrieved context.
Detects hallucinated claims not backed by source documents.
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger


@dataclass
class GroundingClaim:
    """A single claim extracted from the LLM response."""
    claim_text: str
    is_grounded: bool
    supporting_chunk: Optional[str] = None
    confidence: float = 0.0


@dataclass
class GroundingReport:
    """Full grounding verification report for a response."""
    response_text: str
    total_claims: int
    grounded_claims: int
    ungrounded_claims: int
    grounding_score: float  # 0.0 to 1.0
    claims: List[GroundingClaim]
    has_citations: bool
    citation_count: int
    verified_at: str = ""

    def __post_init__(self):
        if not self.verified_at:
            self.verified_at = datetime.utcnow().isoformat()


def extract_claims(response_text: str) -> List[str]:
    """Extract verifiable claims from a response (sentence-level)."""
    sentences = re.split(r'(?<=[.!?])\s+', response_text)

    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        skip_patterns = [
            r'^(I |Let me|Here|Sure|Of course|Based on|According)',
            r'^(Yes|No|However|Additionally|Furthermore|In summary)',
            r'(I\'m not sure|I cannot|I don\'t have)',
        ]
        is_meta = False
        for pat in skip_patterns:
            if re.match(pat, sent, re.IGNORECASE):
                is_meta = True
                break

        if not is_meta or len(sent) > 50:
            claims.append(sent)

    return claims


def check_claim_against_context(
    claim: str,
    context_chunks: List[str],
    threshold: float = 0.3,
) -> Tuple[bool, Optional[str], float]:
    """
    Check if a claim is supported by any context chunk.
    Uses keyword overlap as a simple grounding heuristic.

    Returns: (is_grounded, supporting_chunk_or_none, confidence)
    """
    claim_lower = claim.lower()
    claim_words = set(re.findall(r'\b\w{3,}\b', claim_lower))

    if not claim_words:
        return True, None, 1.0

    best_overlap = 0.0
    best_chunk = None

    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_lower))

        if not chunk_words:
            continue

        overlap = len(claim_words & chunk_words) / len(claim_words)

        key_phrases = re.findall(r'\b\w+\s+\w+(?:\s+\w+)?\b', claim_lower)
        phrase_found = any(phrase in chunk_lower for phrase in key_phrases if len(phrase) > 8)

        if phrase_found:
            overlap = max(overlap, 0.6)

        claim_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
        if claim_numbers:
            chunk_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', chunk))
            if claim_numbers & chunk_numbers:
                overlap = max(overlap, 0.5)
            elif claim_numbers and not (claim_numbers & chunk_numbers):
                overlap = min(overlap, 0.2)

        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk

    is_grounded = best_overlap >= threshold
    return is_grounded, best_chunk if is_grounded else None, best_overlap


def verify_grounding(
    response_text: str,
    context_chunks: List[str],
    threshold: float = 0.3,
) -> GroundingReport:
    """
    Verify that an LLM response is grounded in the retrieved context.

    Args:
        response_text: The LLM's response
        context_chunks: List of retrieved document chunks
        threshold: Minimum overlap score to consider a claim grounded

    Returns:
        GroundingReport with claim-level verification
    """
    claims_text = extract_claims(response_text)

    if not claims_text:
        return GroundingReport(
            response_text=response_text,
            total_claims=0,
            grounded_claims=0,
            ungrounded_claims=0,
            grounding_score=1.0,
            claims=[],
            has_citations=False,
            citation_count=0,
        )

    verified_claims = []
    for claim_text in claims_text:
        is_grounded, supporting, confidence = check_claim_against_context(
            claim_text, context_chunks, threshold
        )
        verified_claims.append(GroundingClaim(
            claim_text=claim_text,
            is_grounded=is_grounded,
            supporting_chunk=supporting[:200] if supporting else None,
            confidence=confidence,
        ))

    grounded = sum(1 for c in verified_claims if c.is_grounded)
    ungrounded = len(verified_claims) - grounded

    citation_patterns = [
        r'\[.*?\]',
        r'(?:according to|per|as stated in|as described in)',
        r'(?:source:|reference:|see:)',
    ]
    citation_count = 0
    for pattern in citation_patterns:
        citation_count += len(re.findall(pattern, response_text, re.IGNORECASE))

    grounding_score = grounded / len(verified_claims) if verified_claims else 1.0

    return GroundingReport(
        response_text=response_text,
        total_claims=len(verified_claims),
        grounded_claims=grounded,
        ungrounded_claims=ungrounded,
        grounding_score=round(grounding_score, 3),
        claims=verified_claims,
        has_citations=citation_count > 0,
        citation_count=citation_count,
    )


def format_grounding_report(report: GroundingReport) -> str:
    """Format grounding report as human-readable text."""
    lines = [
        f"=== Grounding Verification Report ===",
        f"Grounding Score: {report.grounding_score:.1%}",
        f"Total Claims: {report.total_claims}",
        f"Grounded: {report.grounded_claims}",
        f"Ungrounded: {report.ungrounded_claims}",
        f"Citations Found: {report.citation_count}",
        "",
    ]

    if report.claims:
        lines.append("--- Claim Details ---")
        for i, claim in enumerate(report.claims, 1):
            status = "[OK]" if claim.is_grounded else "[UNGROUNDED]"
            lines.append(f"{status} Claim {i} (confidence: {claim.confidence:.2f}):")
            lines.append(
                f"   \"{claim.claim_text[:120]}...\"" if len(claim.claim_text) > 120
                else f"   \"{claim.claim_text}\""
            )
            if claim.supporting_chunk:
                lines.append(f"   Supported by: \"{claim.supporting_chunk[:100]}...\"")
        lines.append("")

    return "\n".join(lines)
