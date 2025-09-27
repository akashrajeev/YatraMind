# backend/app/security.py
from __future__ import annotations

from fastapi import Header, HTTPException
from app.config import settings
from app.services.auth_service import get_current_user, require_permission, require_role
from app.models.user import User

async def require_api_key(x_api_key: str | None = Header(default=None)):
    """Simple API key authentication (for backward compatibility)"""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Re-export auth functions for convenience
__all__ = [
    "require_api_key",
    "get_current_user", 
    "require_permission",
    "require_role"
]
