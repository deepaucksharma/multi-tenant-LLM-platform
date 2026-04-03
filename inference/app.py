"""
Main FastAPI inference server.
Multi-tenant LLM inference with streaming, RAG, tenant routing,
canary deployment, audit logging, and feedback collection.

Usage:
    uvicorn inference.app:app --host 0.0.0.0 --port 8000
    PRELOAD_MODEL=true uvicorn inference.app:app --host 0.0.0.0 --port 8000
"""
import os
import time
import json
import asyncio
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from inference.schemas import (
    ChatRequest, ChatResponse, Citation, StreamChunk,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, TenantInfo, ModelRegistryEntry,
    CanaryConfig,
)
from inference.auth import get_auth_context, verify_tenant_access, AuthContext
from inference.adapter_manager import get_adapter_manager
from inference.model_backend import get_model_backend
from inference.tenant_router import get_tenant_route, list_tenants, validate_tenant_isolation
from inference.canary import get_canary_manager
from inference.audit_logger import get_audit_logger

# Track server start time
_server_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup."""
    logger.info("Starting inference server...")

    preload = os.getenv("PRELOAD_MODEL", "false").lower() == "true"
    if preload:
        logger.info("Pre-loading inference backend...")
        backend = get_model_backend()
        backend.warmup()
        logger.info("Inference backend pre-loaded")
    else:
        logger.info("Model will be loaded on first request (set PRELOAD_MODEL=true to preload)")

    yield

    logger.info("Shutting down inference server...")


app = FastAPI(
    title="Multi-Tenant LLM Inference API",
    description="Domain-adapted LLM serving with tenant isolation, RAG, and streaming",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health & Info Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    backend = get_model_backend()
    stats = backend.get_stats()
    base_loaded = bool(stats.get("base_loaded"))
    ready = bool(stats.get("ready", base_loaded))
    if base_loaded:
        status = "healthy"
    elif ready and stats.get("load_strategy") == "lazy":
        status = "ready"
    elif ready:
        status = "healthy"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        gpu_available=bool(stats.get("gpu_memory", {}).get("available", True)),
        gpu_memory_gb=stats.get("gpu_memory", {}).get("allocated_gb"),
        models_loaded=[stats.get("backend", "base")] if base_loaded else [],
        adapters_available={
            key: ["configured"]
            for key in stats.get("available_adapters", [])
        },
        uptime_seconds=round(time.time() - _server_start_time, 1),
    )


@app.get("/backend/status", tags=["System"])
async def backend_status():
    """Detailed backend status — includes which backend is active and resolved models."""
    backend = get_model_backend()
    return backend.get_stats()


@app.get("/tenants", tags=["System"])
async def get_tenants():
    """List all available tenants."""
    return list_tenants()


@app.get("/tenants/{tenant_id}", response_model=TenantInfo, tags=["System"])
async def get_tenant_info(tenant_id: str):
    """Get detailed tenant information."""
    try:
        route = get_tenant_route(tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    doc_count = 0
    try:
        from rag.build_index import get_chroma_client, get_collection_name
        client = get_chroma_client()
        collection = client.get_collection(get_collection_name(tenant_id))
        doc_count = collection.count()
    except Exception:
        pass

    return TenantInfo(
        tenant_id=route.tenant_id,
        domain=route.domain,
        active_model_type=route.model_type,
        adapter_path=route.adapter_key,
        rag_collection=route.rag_collection,
        document_count=doc_count,
        topics=route.topics,
    )


# ============================================================
# Chat Endpoint (non-streaming)
# ============================================================

@app.post("/chat", response_model=ChatResponse, tags=["Inference"])
async def chat(
    request: ChatRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Send a chat message and receive a complete response.
    Supports RAG, tenant-specific routing, and audit logging.
    """
    verify_tenant_access(auth, request.tenant_id.value)
    tenant_id = request.tenant_id.value
    t_start = time.time()

    try:
        # Get routing config
        canary_mgr = get_canary_manager()
        model_type, is_canary = canary_mgr.get_model_type(tenant_id)
        if request.model_type.value != "sft":
            model_type = request.model_type.value

        route = get_tenant_route(tenant_id, model_type)

        backend = get_model_backend()
        backend.prepare_route(route)
        model_version = backend.get_model_label(tenant_id, model_type, route)

        # Build messages
        messages = [{"role": "system", "content": route.system_prompt}]
        for msg in request.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # RAG retrieval
        citations = []
        retrieval_time_ms = 0
        retrieval_method = "none"
        grounding_score = None

        if request.use_rag:
            try:
                from rag.rag_chain import RAGRequest, execute_rag_chain

                rag_request = RAGRequest(
                    query=request.message,
                    tenant_id=tenant_id,
                    top_k=3,
                    topic_filter=request.topic_filter,
                )

                def gen_fn(msgs):
                    return backend.generate(
                        msgs,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        tenant_id=tenant_id,
                        model_type=model_type,
                        adapter_key=route.adapter_key,
                    )

                rag_response = execute_rag_chain(rag_request, generate_fn=gen_fn)

                # Enforce tenant isolation: filter any cross-tenant chunks before
                # they are surfaced as citations or influence downstream grounding.
                clean_citations = validate_tenant_isolation(
                    tenant_id,
                    rag_response.citations,
                )

                answer = rag_response.answer
                citations = [Citation(**c) for c in clean_citations]
                retrieval_time_ms = rag_response.retrieval_time_ms
                retrieval_method = rag_response.retrieval_result.get("retrieval_method", "")
                if rag_response.grounding_report:
                    grounding_score = rag_response.grounding_report.get("grounding_score")

            except Exception as e:
                logger.warning(f"RAG failed, falling back to direct generation: {e}")
                messages.append({"role": "user", "content": request.message})
                answer = backend.generate(
                    messages,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    tenant_id=tenant_id,
                    model_type=model_type,
                    adapter_key=route.adapter_key,
                )
        else:
            messages.append({"role": "user", "content": request.message})
            answer = backend.generate(
                messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                tenant_id=tenant_id,
                model_type=model_type,
                adapter_key=route.adapter_key,
            )

        total_time = round((time.time() - t_start) * 1000, 2)
        generation_time = total_time - retrieval_time_ms

        # Audit log
        audit = get_audit_logger()
        request_id = audit.log_request(
            tenant_id=tenant_id,
            user_message=request.message,
            model_response=answer,
            model_type=model_type,
            model_version=model_version,
            adapter_key=route.adapter_key,
            is_canary=is_canary,
            use_rag=request.use_rag,
            retrieval_method=retrieval_method,
            citations_count=len(citations),
            grounding_score=grounding_score,
            latency_ms=total_time,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time,
            token_count=len(answer.split()),
        )

        return ChatResponse(
            tenant_id=tenant_id,
            message=answer,
            citations=citations,
            model_version=model_version,
            model_type=model_type,
            retrieval_method=retrieval_method,
            grounding_score=grounding_score,
            latency_ms=total_time,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        audit = get_audit_logger()
        audit.log_request(
            tenant_id=tenant_id,
            user_message=request.message,
            model_response="",
            error=str(e),
            latency_ms=round((time.time() - t_start) * 1000, 2),
        )
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Streaming Chat Endpoint
# ============================================================

@app.post("/chat/stream", tags=["Inference"])
async def chat_stream(
    request: ChatRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Send a chat message and receive a streaming SSE response.
    Tokens are sent as they are generated.
    """
    verify_tenant_access(auth, request.tenant_id.value)
    tenant_id = request.tenant_id.value

    async def event_generator():
        t_start = time.time()
        full_response = ""

        try:
            canary_mgr = get_canary_manager()
            model_type, is_canary = canary_mgr.get_model_type(tenant_id)
            if request.model_type.value != "sft":
                model_type = request.model_type.value

            route = get_tenant_route(tenant_id, model_type)

            backend = get_model_backend()
            backend.prepare_route(route)
            model_version = backend.get_model_label(tenant_id, model_type, route)

            messages = [{"role": "system", "content": route.system_prompt}]
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})

            citations_data = []
            retrieval_chunks = []
            retrieval_time_ms = 0

            if request.use_rag:
                try:
                    from rag.retriever import retrieve, format_context_for_llm, format_citations
                    from rag.grounding import verify_grounding

                    retrieval_result = retrieve(
                        query=request.message,
                        tenant_id=tenant_id,
                        top_k=3,
                    )
                    retrieval_time_ms = retrieval_result.retrieval_time_ms

                    # Enforce tenant isolation: filter cross-tenant chunks before
                    # they reach the prompt or are returned as citations.
                    # Note: the retriever already applies _validate_isolation internally;
                    # this is a defense-in-depth second pass at the app layer.
                    clean_chunk_ids = {
                        ch["chunk_id"]
                        for ch in validate_tenant_isolation(
                            tenant_id,
                            [
                                {
                                    "chunk_id": c.citation_key,
                                    "tenant_id": getattr(c, "tenant_id", tenant_id),
                                }
                                for c in retrieval_result.chunks
                            ],
                        )
                    }
                    clean_result_chunks = [
                        c for c in retrieval_result.chunks
                        if c.citation_key in clean_chunk_ids
                    ]
                    retrieval_chunks = [c.content for c in clean_result_chunks]

                    # Build context and citations from the filtered chunk list only
                    from dataclasses import replace as dc_replace
                    filtered_result = dc_replace(
                        retrieval_result,
                        chunks=clean_result_chunks,
                        total_retrieved=len(clean_result_chunks),
                    )
                    context = format_context_for_llm(filtered_result)
                    citations_data = format_citations(filtered_result)

                    user_msg = (
                        f"Context from knowledge base:\n"
                        f"{'='*40}\n{context}\n{'='*40}\n\n"
                        f"Question: {request.message}\n\n"
                        f"Answer based on the context above. Include citations."
                    )
                    messages.append({"role": "user", "content": user_msg})

                except Exception as e:
                    logger.warning(f"RAG failed in stream: {e}")
                    messages.append({"role": "user", "content": request.message})
            else:
                messages.append({"role": "user", "content": request.message})

            # Send citations first
            if citations_data:
                yield {
                    "event": "citations",
                    "data": json.dumps({"citations": citations_data}),
                }

            # Stream tokens
            for token in backend.generate_stream(
                messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                tenant_id=tenant_id,
                model_type=model_type,
                adapter_key=route.adapter_key,
            ):
                full_response += token
                yield {
                    "event": "token",
                    "data": json.dumps({"token": token}),
                }

            total_time = round((time.time() - t_start) * 1000, 2)
            grounding_score = None
            if request.use_rag and retrieval_chunks and full_response:
                try:
                    report = verify_grounding(full_response, retrieval_chunks)
                    grounding_score = report.grounding_score
                except Exception as e:
                    logger.warning(f"Grounding verification failed in stream: {e}")

            # Audit log
            audit = get_audit_logger()
            request_id = audit.log_request(
                tenant_id=tenant_id,
                user_message=request.message,
                model_response=full_response,
                model_type=model_type,
                model_version=model_version,
                adapter_key=route.adapter_key,
                is_canary=is_canary,
                use_rag=request.use_rag,
                citations_count=len(citations_data),
                grounding_score=grounding_score,
                latency_ms=total_time,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=total_time - retrieval_time_ms,
                token_count=len(full_response.split()),
            )

            yield {
                "event": "done",
                "data": json.dumps({
                    "request_id": request_id,
                    "total_tokens": len(full_response.split()),
                    "latency_ms": total_time,
                    "model_version": model_version,
                    "model_type": model_type,
                    "is_canary": is_canary,
                }),
            }

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


# ============================================================
# Feedback Endpoint
# ============================================================

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """Submit feedback on a model response."""
    verify_tenant_access(auth, request.tenant_id.value)

    audit = get_audit_logger()
    feedback_id = audit.log_feedback(
        request_id=request.request_id,
        tenant_id=request.tenant_id.value,
        rating=request.rating,
        feedback_type=request.feedback_type,
        comment=request.comment,
        flagged_issues=request.flagged_issues,
    )

    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


# ============================================================
# Monitoring Endpoints
# ============================================================

@app.get("/stats", tags=["Monitoring"])
async def get_stats(
    tenant_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=720),
):
    """Get request statistics."""
    audit = get_audit_logger()
    return audit.get_request_stats(tenant_id, hours)


@app.get("/stats/recent", tags=["Monitoring"])
async def get_recent(
    tenant_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """Get recent requests."""
    audit = get_audit_logger()
    return audit.get_recent_requests(tenant_id, limit)


@app.get("/model/stats", tags=["Monitoring"])
async def get_model_stats():
    """Get model/adapter statistics. Proxies to the active backend."""
    backend = get_model_backend()
    return backend.get_stats()


# ============================================================
# Canary Endpoints
# ============================================================

@app.post("/canary/configure", tags=["Canary"])
async def configure_canary(config: CanaryConfig):
    """Configure canary deployment for a tenant."""
    canary = get_canary_manager()
    canary.set_canary(
        tenant_id=config.tenant_id.value,
        stable_model=config.stable_model.value,
        canary_model=config.canary_model.value,
        canary_percentage=config.canary_percentage,
        enabled=config.enabled,
    )
    return {"status": "configured", "config": canary.get_stats(config.tenant_id.value)}


@app.post("/canary/disable/{tenant_id}", tags=["Canary"])
async def disable_canary(tenant_id: str):
    """Disable canary (rollback to stable)."""
    canary = get_canary_manager()
    canary.disable_canary(tenant_id)
    return {"status": "disabled", "tenant_id": tenant_id}


@app.post("/canary/promote/{tenant_id}", tags=["Canary"])
async def promote_canary(tenant_id: str):
    """Promote canary to stable (full rollout)."""
    canary = get_canary_manager()
    canary.promote_canary(tenant_id)
    return {"status": "promoted", "tenant_id": tenant_id}


@app.get("/canary/stats", tags=["Canary"])
async def canary_stats(tenant_id: Optional[str] = None):
    """Get canary deployment statistics."""
    canary = get_canary_manager()
    return canary.get_stats(tenant_id)


# ============================================================
# Model Registry
# ============================================================

@app.get("/registry", tags=["Registry"])
async def get_registry(tenant_id: Optional[str] = None):
    """Get model registry entries."""
    from training.mlflow_utils import ModelRegistry
    registry = ModelRegistry()
    models = registry.list_models(tenant_id)
    return {
        "models": models,
        "summary": registry.get_summary(),
    }


# ============================================================
# RAG Debug Endpoint
# ============================================================

@app.get("/rag/test", tags=["Debug"])
async def test_rag(
    query: str = Query(..., description="Search query"),
    tenant_id: str = Query(..., description="Tenant ID"),
    top_k: int = Query(3, ge=1, le=10),
):
    """Test RAG retrieval without LLM generation."""
    try:
        from rag.retriever import retrieve, format_citations

        result = retrieve(query, tenant_id, top_k=top_k)
        return {
            "query": query,
            "tenant_id": tenant_id,
            "total_retrieved": result.total_retrieved,
            "retrieval_method": result.retrieval_method,
            "retrieval_time_ms": result.retrieval_time_ms,
            "chunks": [
                {
                    "rank": c.rank,
                    "score": round(c.score, 3),
                    "title": c.title,
                    "topic": c.topic,
                    "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    "citation_key": c.citation_key,
                }
                for c in result.chunks
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("INFERENCE_PORT", 8000))
    uvicorn.run(
        "inference.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
