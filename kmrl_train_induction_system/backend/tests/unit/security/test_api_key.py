from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app import security


@pytest.mark.asyncio
async def test_api_key_dependency_fails_closed_when_unconfigured(monkeypatch):
    settings = MagicMock(api_key=None)
    monkeypatch.setattr(security, "settings", settings)
    with pytest.raises(HTTPException) as exc:
        await security.require_api_key("anything")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_api_key_dependency_rejects_wrong_key(monkeypatch):
    settings = MagicMock(api_key="expected")
    monkeypatch.setattr(security, "settings", settings)
    with pytest.raises(HTTPException) as exc:
        await security.require_api_key("wrong")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_api_key_dependency_accepts_configured_key(monkeypatch):
    settings = MagicMock(api_key="expected")
    monkeypatch.setattr(security, "settings", settings)
    assert await security.require_api_key("expected") is True
