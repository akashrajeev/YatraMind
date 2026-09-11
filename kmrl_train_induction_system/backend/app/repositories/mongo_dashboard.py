"""Mongo-backed dashboard read repository."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.repositories.protocols import DashboardRepository
from app.utils.cloud_database import cloud_db_manager


class MongoDashboardRepository(DashboardRepository):
    """Encapsulate dashboard aggregation queries and read models."""

    async def _collection(self, name: str):
        return await cloud_db_manager.get_collection(name)

    async def list_trainsets(self) -> Sequence[Mapping[str, Any]]:
        cursor = (await self._collection("trainsets")).find({})
        result: list[Mapping[str, Any]] = []
        async for document in cursor:
            item = dict(document)
            item.pop("_id", None)
            result.append(item)
        return result

    async def get_latest_induction(self) -> Mapping[str, Any] | None:
        document = await (await self._collection("latest_induction")).find_one(
            sort=[("created_at", -1)]
        )
        if document is None:
            return None
        item = dict(document)
        item.pop("_id", None)
        return item

    async def count_pending_assignments(self) -> int:
        return await (await self._collection("assignments")).count_documents({"status": "PENDING"})

    async def list_assigned_trainset_ids(self) -> set[str]:
        cursor = (await self._collection("assignments")).find(
            {"status": {"$in": ["PENDING", "APPROVED", "OVERRIDDEN"]}},
            {"trainset_id": 1},
        )
        result: set[str] = set()
        async for document in cursor:
            trainset_id = document.get("trainset_id")
            if trainset_id:
                result.add(str(trainset_id))
        return result

    async def list_alert_candidates(self) -> Sequence[Mapping[str, Any]]:
        cursor = (await self._collection("trainsets")).find(
            {},
            {
                "trainset_id": 1,
                "fitness_certificates": 1,
                "job_cards": 1,
                "current_mileage": 1,
                "max_mileage_before_maintenance": 1,
            },
        )
        result: list[Mapping[str, Any]] = []
        async for document in cursor:
            item = dict(document)
            item.pop("_id", None)
            result.append(item)
        return result

    async def get_recent_optimization_history(self, limit: int = 7) -> Sequence[Mapping[str, Any]]:
        cursor = (await self._collection("optimization_history")).find().sort("timestamp", -1).limit(limit)
        result: list[Mapping[str, Any]] = []
        async for document in cursor:
            item = dict(document)
            item.pop("_id", None)
            result.append(item)
        return result
