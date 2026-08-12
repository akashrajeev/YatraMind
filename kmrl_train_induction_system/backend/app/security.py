from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import settings
from app.services.auth_service import get_current_user, require_permission, require_role


async def require_api_key(x_api_key: str | None = Header(default=None)) -> bool:
    """Validate the legacy machine-to-machine API key.

    This path is retained for compatibility with older integrations. It fails
    closed when no key is configured rather than accidentally allowing access.
    User/session authentication remains separate via ``auth_service``.
    """
    configured = settings.api_key
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key authentication is not configured",
        )
    if x_api_key != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return True


__all__ = ["require_api_key", "get_current_user", "require_permission", "require_role"]
