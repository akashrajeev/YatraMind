"""Compatibility wrappers for optimization history access.

Persistence is implemented by repository adapters; these functions remain as
thin wrappers so existing API callers can migrate without a breaking change.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.repositories.mongo_optimization import MongoOptimizationRepository
from app.utils.cloud_database import cloud_db_manager

_repository = MongoOptimizationRepository()

async def get_latest_decisions() -> Optional[List[Dict[str, Any]]]:
    try:
        # Keep the legacy module-level DB seam patchable for existing tests and
        # integrations while delegating storage behavior to the repository.
        collection = await cloud_db_manager.get_collection("latest_induction")
        document = await collection.find_one(sort=[("created_at", -1)])
        if document and document.get("decisions"):
            return [dict(item) for item in document["decisions"]]
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
