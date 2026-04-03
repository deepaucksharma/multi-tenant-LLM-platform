"""
Pydantic schemas for the inference API.
Defines request/response models for all endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class TenantID(str, Enum):
    SIS = "sis"
    MFG = "mfg"


class ModelType(str, Enum):
    BASE = "base"
    SFT = "sft"
    DPO = "dpo"


# ---- Chat ----

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    tenant_id: TenantID = Field(..., description="Tenant identifier")
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list, description="Previous conversation turns"
    )
    use_rag: bool = Field(default=True, description="Whether to use RAG retrieval")
    use_streaming: bool = Field(default=True, description="Whether to stream response")
    model_type: ModelType = Field(default=ModelType.SFT, description="Model variant to use")
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    topic_filter: Optional[str] = Field(default=None, description="Optional topic filter for RAG")


class Citation(BaseModel):
    title: str
    topic: str
    source_file: str
    relevance_score: float
    citation_key: str


class ChatResponse(BaseModel):
    tenant_id: str
    message: str
    citations: List[Citation] = Field(default_factory=list)
    model_version: str = ""
    model_type: str = ""
    retrieval_method: str = ""
    grounding_score: Optional[float] = None
    latency_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    request_id: str = ""
    timestamp: str = ""


class StreamChunk(BaseModel):
    """SSE stream chunk."""
    token: str = ""
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None


# ---- Feedback ----

class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="Original request ID")
    tenant_id: TenantID
    rating: int = Field(..., ge=1, le=5, description="1=bad, 5=great")
    feedback_type: str = Field(default="thumbs", description="thumbs, detailed, flag")
    comment: Optional[str] = None
    flagged_issues: List[str] = Field(
        default_factory=list,
        description="Issues: hallucination, wrong_domain, unsafe, unhelpful, pii_leak"
    )


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


# ---- Health / Info ----

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    models_loaded: List[str] = Field(default_factory=list)
    adapters_available: Dict[str, List[str]] = Field(default_factory=dict)
    uptime_seconds: float = 0.0


class TenantInfo(BaseModel):
    tenant_id: str
    domain: str
    active_model_type: str
    adapter_path: str
    rag_collection: str
    document_count: int
    topics: List[str]


class ModelRegistryEntry(BaseModel):
    id: str
    tenant_id: str
    model_type: str
    version: str
    status: str
    metrics: Dict[str, Any]
    registered_at: str


# ---- Canary ----

class CanaryConfig(BaseModel):
    tenant_id: TenantID
    stable_model: ModelType = Field(default=ModelType.SFT)
    canary_model: ModelType = Field(default=ModelType.DPO)
    canary_percentage: float = Field(default=10.0, ge=0.0, le=100.0)
    enabled: bool = Field(default=False)
