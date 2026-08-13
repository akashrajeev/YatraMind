"""Mongo-backed read model repository for dashboard queries."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

from app.models.assignment import AssignmentStatus
from app.repositories.protocols import DashboardRepository
from app.utils.cloud_database import cloud_db_manager


class MongoDashboardRepository(DashboardRepository):
    async def list_trainsets(self) -> Sequence[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection("trainsets")
        result: list[Mapping[str, Any]] = []
        async for document in collection.find({}):
            item = dict(document)
            item.pop("_id", None)
            result.append(item)
        return result

    async def get_latest_induction(self) -> Mapping[str, Any] | None:
        collection = await cloud_db_manager.get_collection("latest_induction")
        return await collection.find_one(sort=[("_meta.updated_at", -1), ("created_at", -1)])

    async def count_pending_assignments(self) -> int:
        collection = await cloud_db_manager.get_collection("assignments")
        return await collection.count_documents({"status": AssignmentStatus.PENDING.value})

    async def list_assigned_trainset_ids(self) -> set[str]:
        collection = await cloud_db_manager.get_collection("assignments")
        assigned: set[str] = set()
        cursor = collection.find(
            {"status": {"$in": [
                AssignmentStatus.PENDING.value,
                AssignmentStatus.APPROVED.value,
                AssignmentStatus.OVERRIDDEN.value,
            ]}},
            {"trainset_id": 1},
        )
        async for document in cursor:
            trainset_id = document.get("trainset_id")
            if trainset_id:
                assigned.add(str(trainset_id))
        return assigned

    async def get_recent_optimization_history(self, limit: int = 7) -> Sequence[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection("optimization_history")
        cursor = collection.find().sort("timestamp", -1).limit(limit)
        history: list[Mapping[str, Any]] = []
        async for document in cursor:
            item = dict(document)
            item.pop("_id", None)
            history.append(item)
        return history

    async def list_alert_candidates(self) -> Sequence[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection("trainsets")
        pipeline = [
            {"$addFields": {"certs_array": {"$objectToArray": "$fitness_certificates"}}},
            {"$match": {"$or": [
                {"certs_array.v.status": {"$in": ["EXPIRED", "EXPIRING_SOON"]}},
                {"job_cards.critical_cards": {"$gt": 0}},
                {"$expr": {"$gte": ["$current_mileage", {"$multiply": ["$max_mileage_before_maintenance", 0.95]}]}},
            ]}},
            {"$project": {
                "trainset_id": 1,
                "fitness_certificates": 1,
                "job_cards": 1,
                "current_mileage": 1,
                "max_mileage_before_maintenance": 1,
            }},
        ]
        candidates: list[Mapping[str, Any]] = []
        async for document in collection.aggregate(pipeline):
            candidates.append(dict(document))
        return candidates
