"""Compatibility wrappers for optimization history access.

Persistence is implemented by repository adapters; these functions remain as
thin wrappers so existing API callers can migrate without a breaking change.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from app.repositories.mongo_optimization import MongoOptimizationRepository

_repository = MongoOptimizationRepository()

async def get_latest_decisions() -> Optional[List[Dict[str, Any]]]:
    try:
        decisions = await _repository.get_latest_decisions()
        return [dict(item) for item in decisions] if decisions else None
    except Exception:
        return None

async def get_decisions_from_history() -> Optional[List[Dict[str, Any]]]:
    try:
        document = await _repository.get_latest_history()
        decisions = document.get("decisions") if document else None
        return [dict(item) for item in decisions] if isinstance(decisions, list) and decisions else None
    except Exception:
        return None
