"""Mongo-backed model/policy status repository."""
from __future__ import annotations

from typing import Any

from app.utils.cloud_database import cloud_db_manager


class MongoPolicyRepository:
    """Read the latest registered AI model metadata for policy-status views."""

    COLLECTIONS = {
        "failure_risk": "failure_risk_models",
        "service_selection": "service_selection_models",
        "rl_stabling": "rl_stabling_policies",
    }

    async def latest_model_status(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for key, collection_name in self.COLLECTIONS.items():
            collection = await cloud_db_manager.get_collection(collection_name)
            document = await collection.find_one(sort=[("meta.created_at", -1)])
            meta = (document or {}).get("meta", {})
            result[key] = {
                "loaded": document is not None,
                "version": meta.get("version"),
                "created_at": meta.get("created_at"),
            }
        return result
