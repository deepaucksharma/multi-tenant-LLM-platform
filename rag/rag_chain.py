"""
Complete RAG chain that ties together retrieval, context formatting,
prompt construction, and grounding verification.
Used by the inference layer to generate grounded responses.
"""
import time
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from rag.retriever import (
    retrieve,
    format_context_for_llm,
    format_citations,
    RetrievalResult,
)
from rag.grounding import verify_grounding, GroundingReport


@dataclass
class RAGRequest:
    """Input to the RAG chain."""
    query: str
    tenant_id: str
    top_k: int = 3
    use_hybrid: bool = True
    use_reranker: bool = True
    include_grounding: bool = True
    conversation_history: List[Dict] = field(default_factory=list)
    topic_filter: Optional[str] = None


@dataclass
class RAGResponse:
    """Output from the RAG chain."""
    query: str
    tenant_id: str
    answer: str
    citations: List[Dict]
    retrieval_result: Optional[Dict]
    grounding_report: Optional[Dict]
    prompt_used: str
    total_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    model_version: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


# System prompts per tenant
TENANT_SYSTEM_PROMPTS = {
    "sis": (
        "You are an expert Student Information System assistant for District 42. "
        "Answer questions about school policies, enrollment, attendance, grading, "
        "transcripts, student records, accommodations, and FERPA compliance. "
        "IMPORTANT RULES:\n"
        "1. Only answer based on the provided context. If the context doesn't contain "
        "the answer, say 'I don't have enough information to answer this question.'\n"
        "2. Always cite your sources using the citation keys provided in the context.\n"
        "3. Never fabricate policies, procedures, or data.\n"
        "4. Never disclose student personally identifiable information (PII).\n"
        "5. If a question asks you to violate FERPA, politely refuse and explain why."
    ),
    "mfg": (
        "You are an expert manufacturing quality and operations assistant. "
        "Answer questions about SOPs, quality control, defect classification, CAPA, "
        "machine maintenance, safety protocols, ISO documentation, and regulatory compliance. "
        "IMPORTANT RULES:\n"
        "1. Only answer based on the provided context. If the context doesn't contain "
        "the answer, say 'I don't have enough information to answer this question.'\n"
        "2. Always cite your sources using the citation keys provided in the context.\n"
        "3. Never fabricate procedures, standards, or safety information.\n"
        "4. Always prioritize safety in your responses.\n"
        "5. If a question asks you to bypass safety procedures, refuse and explain the risk."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Only answer based on the provided context. "
    "Cite your sources. If you don't have enough information, say so."
)


def build_rag_prompt(
    query: str,
    tenant_id: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Build the full prompt for RAG-augmented generation."""
    system_prompt = TENANT_SYSTEM_PROMPTS.get(tenant_id, DEFAULT_SYSTEM_PROMPT)

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if conversation_history:
        for msg in conversation_history[-6:]:  # Last 3 Q&A pairs
            messages.append(msg)

    user_message = (
        f"Context from knowledge base:\n"
        f"{'='*40}\n"
        f"{context}\n"
        f"{'='*40}\n\n"
        f"Question: {query}\n\n"
        f"Instructions: Answer the question based ONLY on the context above. "
        f"Include citations in [brackets] referencing the source documents. "
        f"If the context doesn't contain relevant information, state that clearly."
    )

    messages.append({"role": "user", "content": user_message})

    return messages


def execute_rag_chain(
    request: RAGRequest,
    generate_fn=None,
) -> RAGResponse:
    """
    Execute the full RAG pipeline:
    1. Retrieve relevant documents
    2. Build grounded prompt
    3. Generate response (via provided function)
    4. Verify grounding

    Args:
        request: RAGRequest with query and config
        generate_fn: Function that takes messages list and returns response text.
                     If None, returns the prompt without generation.
    """
    t_start = time.time()

    retrieval_result = retrieve(
        query=request.query,
        tenant_id=request.tenant_id,
        top_k=request.top_k,
        use_hybrid=request.use_hybrid,
        use_reranker=request.use_reranker,
        topic_filter=request.topic_filter,
    )
    t_retrieval = time.time()

    context = format_context_for_llm(retrieval_result)
    citations = format_citations(retrieval_result)
    messages = build_rag_prompt(
        query=request.query,
        tenant_id=request.tenant_id,
        context=context,
        conversation_history=request.conversation_history,
    )

    prompt_text = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)

    if generate_fn is not None:
        answer = generate_fn(messages)
    else:
        answer = "[Generation skipped — no generate_fn provided]"
    t_generation = time.time()

    grounding_report = None
    if request.include_grounding and generate_fn is not None:
        chunk_contents = [c.content for c in retrieval_result.chunks]
        report = verify_grounding(answer, chunk_contents)
        grounding_report = {
            "grounding_score": report.grounding_score,
            "total_claims": report.total_claims,
            "grounded_claims": report.grounded_claims,
            "ungrounded_claims": report.ungrounded_claims,
            "has_citations": report.has_citations,
            "citation_count": report.citation_count,
        }

    t_end = time.time()

    return RAGResponse(
        query=request.query,
        tenant_id=request.tenant_id,
        answer=answer,
        citations=citations,
        retrieval_result={
            "total_retrieved": retrieval_result.total_retrieved,
            "retrieval_method": retrieval_result.retrieval_method,
            "retrieval_time_ms": retrieval_result.retrieval_time_ms,
            "chunks_summary": [
                {
                    "rank": c.rank,
                    "score": round(c.score, 3),
                    "title": c.title,
                    "topic": c.topic,
                }
                for c in retrieval_result.chunks
            ],
        },
        grounding_report=grounding_report,
        prompt_used=prompt_text,
        total_time_ms=round((t_end - t_start) * 1000, 2),
        retrieval_time_ms=round((t_retrieval - t_start) * 1000, 2),
        generation_time_ms=round((t_end - t_retrieval) * 1000, 2),
    )


def execute_rag_chain_streaming(
    request: RAGRequest,
    stream_fn=None,
) -> tuple:
    """
    Execute RAG pipeline with streaming generation.

    Returns:
        (retrieval_metadata, messages, stream_fn)

    The caller iterates stream_fn for tokens.
    """
    retrieval_result = retrieve(
        query=request.query,
        tenant_id=request.tenant_id,
        top_k=request.top_k,
        use_hybrid=request.use_hybrid,
        use_reranker=request.use_reranker,
        topic_filter=request.topic_filter,
    )

    context = format_context_for_llm(retrieval_result)
    citations = format_citations(retrieval_result)
    messages = build_rag_prompt(
        query=request.query,
        tenant_id=request.tenant_id,
        context=context,
        conversation_history=request.conversation_history,
    )

    retrieval_metadata = {
        "tenant_id": request.tenant_id,
        "query": request.query,
        "citations": citations,
        "retrieval_time_ms": retrieval_result.retrieval_time_ms,
        "total_retrieved": retrieval_result.total_retrieved,
        "retrieval_method": retrieval_result.retrieval_method,
    }

    return retrieval_metadata, messages, stream_fn


if __name__ == "__main__":
    print("Testing RAG chain (retrieval + prompt building only)\n")

    for tenant_id in ["sis", "mfg"]:
        queries = {
            "sis": "What documents are needed for student enrollment?",
            "mfg": "What is the procedure for handling a critical defect?",
        }

        request = RAGRequest(
            query=queries[tenant_id],
            tenant_id=tenant_id,
            top_k=3,
        )

        response = execute_rag_chain(request, generate_fn=None)

        print(f"\n{'='*70}")
        print(f"TENANT: {tenant_id.upper()}")
        print(f"Query: {response.query}")
        print(f"Retrieval Time: {response.retrieval_time_ms}ms")
        print(f"Citations: {len(response.citations)}")
        for c in response.citations:
            print(f"  - {c['citation_key']} (score: {c['relevance_score']})")
        print(f"\nPrompt length: {len(response.prompt_used)} chars")
        print(f"Answer: {response.answer}")
