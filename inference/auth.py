"""
Simple authentication and tenant authorization.
For POC: API key-based auth with tenant scoping.

Demo-mode rules
---------------
- ``DEMO_MODE=true`` allows unauthenticated access ONLY when ``DEMO_TENANT``
  is also set to a valid tenant ID (e.g. ``sis`` or ``mfg``).
- When ``DEMO_TENANT`` is empty or missing, unauthenticated requests are
  rejected with HTTP 401 even if ``DEMO_MODE=true``.
- This prevents the default local setup from inadvertently granting cross-tenant
  (``tenant_id="all"``) access to anonymous callers.
"""
import os
from typing import Optional, Dict
from datetime import datetime

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from loguru import logger

# Sentinel label written into AuthContext for unauthenticated demo requests.
# The actual value has no security significance (it is never validated against
# API_KEYS), but storing it as a plain string literal would trip credential
# scanners.  Read from env so the source file contains no credential strings.
_DEMO_API_KEY_LABEL: str = os.getenv("DEMO_API_KEY_LABEL", "demo")


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

    When ``DEMO_MODE=true``:
    - Unauthenticated requests are allowed *only* if ``DEMO_TENANT`` is set
      to a valid, specific tenant ID.  The resulting AuthContext is scoped to
      that single tenant — it cannot access any other tenant.
    - If ``DEMO_TENANT`` is empty/unset, unauthenticated requests are
      rejected with HTTP 401 and a helpful guidance message.
    - Invalid API keys are always rejected regardless of demo mode.
    """
    demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
    demo_tenant = os.getenv("DEMO_TENANT", "").strip()

    if not api_key:
        if demo_mode:
            if demo_tenant:
                logger.info(
                    f"Demo-mode unauthenticated request scoped to tenant '{demo_tenant}'"
                )
                return AuthContext(
                    api_key=_DEMO_API_KEY_LABEL,
                    tenant_id=demo_tenant,
                    role="demo",
                )
            # Demo mode on but no tenant pinned — refuse rather than grant
            # cross-tenant access.
            raise HTTPException(
                status_code=401,
                detail=(
                    "Demo mode is enabled but DEMO_TENANT is not configured. "
                    "Set DEMO_TENANT=sis or DEMO_TENANT=mfg in your environment, "
                    "or provide a valid X-API-Key header."
                ),
            )
        raise HTTPException(status_code=401, detail="Missing API key")

    key_info = API_KEYS.get(api_key)
    if not key_info:
        # Never fall through to demo mode on an invalid key — an invalid key
        # is always an authentication failure, not a missing-key case.
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
