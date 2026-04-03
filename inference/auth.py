"""
Simple authentication and tenant authorization.
For POC: API key-based auth with tenant scoping.
"""
import os
from typing import Optional, Dict
from datetime import datetime

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from loguru import logger


# Simple API key store (in production: use a proper auth service)
API_KEYS: Dict[str, Dict] = {
    "poc-sis-key-001": {
        "tenant_id": "sis",
        "role": "admin",
        "created_at": "2025-01-01",
    },
    "poc-mfg-key-001": {
        "tenant_id": "mfg",
        "role": "admin",
        "created_at": "2025-01-01",
    },
    "poc-demo-key-all": {
        "tenant_id": "all",  # Can access any tenant (demo mode)
        "role": "demo",
        "created_at": "2025-01-01",
    },
}

# Header for API key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthContext:
    """Authentication context for a request."""

    def __init__(self, api_key: str, tenant_id: str, role: str):
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.role = role
        self.authenticated_at = datetime.utcnow().isoformat()

    def can_access_tenant(self, requested_tenant: str) -> bool:
        """Check if this auth context can access the requested tenant."""
        if self.tenant_id == "all":
            return True
        return self.tenant_id == requested_tenant


async def get_auth_context(
    api_key: Optional[str] = Security(api_key_header),
) -> AuthContext:
    """
    Validate API key and return auth context.
    In demo mode, allows unauthenticated access.
    """
    demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"

    if not api_key:
        if demo_mode:
            return AuthContext(
                api_key="demo",
                tenant_id="all",
                role="demo",
            )
        raise HTTPException(status_code=401, detail="Missing API key")

    key_info = API_KEYS.get(api_key)
    if not key_info:
        if demo_mode:
            return AuthContext(api_key="demo", tenant_id="all", role="demo")
        raise HTTPException(status_code=403, detail="Invalid API key")

    return AuthContext(
        api_key=api_key,
        tenant_id=key_info["tenant_id"],
        role=key_info["role"],
    )


def verify_tenant_access(auth: AuthContext, requested_tenant: str):
    """Verify that the authenticated user can access the requested tenant."""
    if not auth.can_access_tenant(requested_tenant):
        logger.warning(
            f"Tenant access denied: key={auth.api_key[:8]}... "
            f"authorized_tenant={auth.tenant_id} "
            f"requested_tenant={requested_tenant}"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: your key is scoped to tenant '{auth.tenant_id}', "
                   f"but you requested tenant '{requested_tenant}'",
        )
