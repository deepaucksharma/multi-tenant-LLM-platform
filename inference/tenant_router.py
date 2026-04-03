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
# Topics and system prompts are imported from the pipeline config to ensure a single source of truth.
# Policies remain inference-layer concerns and live here.
TENANT_ROUTES: Dict[str, Dict] = {
    "sis": {
        "domain": "Student Information System / Education",
        "system_prompt": _PIPELINE_TENANTS["sis"].system_prompt,
        "rag_collection": "tenant_sis_docs",
        "default_model_type": "sft",
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
        "system_prompt": _PIPELINE_TENANTS["mfg"].system_prompt,
        "rag_collection": "tenant_mfg_docs",
        "default_model_type": "sft",
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

    # Cold-start fix: if the manager hasn't loaded the base model yet,
    # _available_adapters is empty and get_adapter_key() always returns "",
    # causing the first tenant request to run on the bare base model.
    # We trigger model loading here so that _scan_adapters() populates
    # _available_adapters before we commit to the empty adapter key.
    if adapter_key == "" and not manager.is_loaded:
        logger.info(
            f"[{tenant_id}] Cold-start detected — loading base model and "
            f"scanning adapters before resolving route."
        )
        manager.load_base_model()  # also calls _scan_adapters() internally
        # Re-resolve now that _available_adapters is populated.
        adapter_key = manager.get_adapter_key(tenant_id, effective_model_type)
        if adapter_key:
            logger.info(
                f"[{tenant_id}] Post-load adapter resolved: '{adapter_key}'"
            )
        else:
            logger.info(
                f"[{tenant_id}] No trained adapter found for '{effective_model_type}'; "
                f"will serve from base model."
            )

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
