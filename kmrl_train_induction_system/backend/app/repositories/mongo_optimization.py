"""Mongo-backed optimization history repository."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Mapping, Sequence
from app.repositories.protocols import OptimizationRepository
from app.utils.cloud_database import cloud_db_manager

class MongoOptimizationRepository(OptimizationRepository):
    async def save_run(self, payload: Mapping[str, Any]) -> str:
        history = await cloud_db_manager.get_collection("optimization_history")
        result = await history.insert_one(dict(payload))

        latest = await cloud_db_manager.get_collection("latest_induction")
        fleet_requirement = payload.get("fleet_requirement")
        latest_doc = {
            "_meta": {"updated_at": datetime.utcnow().isoformat()},
            "decisions": payload.get("decisions", []),
        }
        if fleet_requirement is not None:
            latest_doc["fleet_requirement"] = fleet_requirement
        await latest.delete_many({})
        await latest.insert_one(latest_doc)
        return str(result.inserted_id)

    async def get_latest_decisions(self) -> Sequence[Mapping[str, Any]] | None:
        collection = await cloud_db_manager.get_collection("latest_induction")
        document = await collection.find_one(sort=[("_meta.updated_at", -1), ("created_at", -1)])
        decisions = document.get("decisions") if document else None
        return decisions if isinstance(decisions, list) and decisions else None

    async def get_latest_history(self) -> Mapping[str, Any] | None:
        collection = await cloud_db_manager.get_collection("optimization_history")
        return await collection.find_one(sort=[("timestamp", -1)])
