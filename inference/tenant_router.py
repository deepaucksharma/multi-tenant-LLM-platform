"""
Tenant-aware request routing.
Routes requests to the correct adapter, system prompt, and RAG collection.
Enforces tenant isolation at every layer.
"""
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from loguru import logger

from inference.adapter_manager import get_adapter_manager
from tenant_data_pipeline.config import TENANTS as _PIPELINE_TENANTS


@dataclass
class TenantRoute:
    """Complete routing configuration for a tenant."""
    tenant_id: str
    domain: str
    system_prompt: str
    adapter_key: str
    rag_collection: str
    model_type: str
    topics: List[str] = field(default_factory=list)
    policies: List[str] = field(default_factory=list)


# Tenant routing table
# Topics are imported from the pipeline config to ensure a single source of truth.
# System prompts and policies are inference-layer concerns and live here.
TENANT_ROUTES: Dict[str, Dict] = {
    "sis": {
        "domain": "Student Information System / Education",
        "system_prompt": (
            "You are an expert Student Information System assistant for District 42. "
            "You help school administrators, teachers, and staff with questions about "
            "enrollment, attendance, grading, transcripts, student records, accommodations, "
            "disciplinary procedures, and FERPA compliance.\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER disclose student personally identifiable information (PII)\n"
            "2. Always comply with FERPA regulations\n"
            "3. Only answer based on the provided knowledge base context\n"
            "4. Include citations from source documents when available\n"
            "5. If you don't have enough information, say so clearly\n"
            "6. If asked about topics outside education/SIS, politely redirect"
        ),
        "rag_collection": "tenant_sis_docs",
        "default_model_type": "sft",
        # Pulled from pipeline config — do NOT duplicate here
        "topics": _PIPELINE_TENANTS["sis"].topics,
        "policies": [
            "FERPA: Never disclose student PII without proper authorization",
            "FERPA: Verify identity before sharing any student information",
            "FERPA: Breach notification within 72 hours",
            "Records: Role-based access control at field level",
        ],
    },
    "mfg": {
        "domain": "Manufacturing / Industrial Quality Control",
        "system_prompt": (
            "You are an expert manufacturing quality and operations assistant. "
            "You help production operators, quality engineers, and maintenance personnel "
            "with questions about SOPs, quality control, defect classification, CAPA processes, "
            "machine maintenance, safety protocols, ISO documentation, and regulatory compliance.\n\n"
            "CRITICAL RULES:\n"
            "1. ALWAYS prioritize safety in your responses\n"
            "2. NEVER suggest bypassing safety procedures or skipping inspections\n"
            "3. Only answer based on the provided knowledge base context\n"
            "4. Include citations from source documents when available\n"
            "5. If you don't have enough information, say so clearly\n"
            "6. If asked about topics outside manufacturing/quality, politely redirect\n"
            "7. When in doubt about safety, recommend consulting a supervisor"
        ),
        "rag_collection": "tenant_mfg_docs",
        "default_model_type": "sft",
        # Pulled from pipeline config — do NOT duplicate here
        "topics": _PIPELINE_TENANTS["mfg"].topics,
        "policies": [
            "Safety: Never bypass LOTO procedures",
            "Safety: All PPE requirements are mandatory",
            "Quality: Never ship non-conforming product without proper disposition",
            "Compliance: All incidents must be reported within 4 hours",
        ],
    },
}


def get_tenant_route(tenant_id: str, model_type: str = None) -> TenantRoute:
    """
    Get the complete routing configuration for a tenant.

    Args:
        tenant_id: Tenant identifier
        model_type: Requested model type (sft, dpo, base)

    Returns:
        TenantRoute with all routing information

    Raises:
        ValueError if tenant not found
    """
    route_config = TENANT_ROUTES.get(tenant_id)
    if not route_config:
        raise ValueError(f"Unknown tenant: {tenant_id}. Available: {list(TENANT_ROUTES.keys())}")

    effective_model_type = model_type or route_config.get("default_model_type", "sft")
    manager = get_adapter_manager()
    adapter_key = manager.get_adapter_key(tenant_id, effective_model_type)

    return TenantRoute(
        tenant_id=tenant_id,
        domain=route_config["domain"],
        system_prompt=route_config["system_prompt"],
        adapter_key=adapter_key,
        rag_collection=route_config["rag_collection"],
        model_type=effective_model_type,
        topics=route_config.get("topics", []),
        policies=route_config.get("policies", []),
    )


def list_tenants() -> List[Dict]:
    """List all available tenants with their configuration."""
    tenants = []
    manager = get_adapter_manager()

    for tenant_id, config in TENANT_ROUTES.items():
        adapter_key = manager.get_adapter_key(tenant_id, config.get("default_model_type", "sft"))
        tenants.append({
            "tenant_id": tenant_id,
            "domain": config["domain"],
            "default_model_type": config.get("default_model_type", "sft"),
            "adapter_available": adapter_key in manager.available_adapters,
            "adapter_key": adapter_key,
            "rag_collection": config["rag_collection"],
            "topics": config.get("topics", []),
        })
    return tenants


def validate_tenant_isolation(tenant_id: str, context_chunks: List[Dict]) -> List[Dict]:
    """
    Validate that retrieved chunks belong to the correct tenant.
    Filter out any cross-tenant contamination.
    """
    clean_chunks = []
    violations = []

    for chunk in context_chunks:
        chunk_tenant = chunk.get("tenant_id", chunk.get("metadata", {}).get("tenant_id", ""))
        if chunk_tenant == tenant_id or chunk_tenant == "":
            clean_chunks.append(chunk)
        else:
            violations.append({
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "expected_tenant": tenant_id,
                "actual_tenant": chunk_tenant,
            })

    if violations:
        logger.warning(
            f"TENANT ISOLATION VIOLATION: {len(violations)} cross-tenant chunks "
            f"filtered for tenant {tenant_id}"
        )

    return clean_chunks
