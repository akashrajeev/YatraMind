"""Mongo-backed assignment repository."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from app.repositories.protocols import AssignmentRepository
from app.utils.cloud_database import cloud_db_manager


class MongoAssignmentRepository(AssignmentRepository):
    """Encapsulate Mongo queries and assignment state transitions."""

    def __init__(self, collection_name: str = "assignments") -> None:
        self.collection_name = collection_name

    async def _collection(self):
        return await cloud_db_manager.get_collection(self.collection_name)

    async def list(
        self,
        *,
        status: str | None = None,
        trainset_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        priority: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Mapping[str, Any]]:
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if trainset_id:
            query["trainset_id"] = trainset_id
        if created_after or created_before:
            created: dict[str, datetime] = {}
            if created_after:
                created["$gte"] = created_after
            if created_before:
                created["$lte"] = created_before
            query["created_at"] = created
        if priority is not None:
            query["priority"] = priority

        cursor = (await self._collection()).find(query).sort("created_at", -1).skip(offset).limit(limit)
        documents: list[Mapping[str, Any]] = []
        async for document in cursor:
            item = dict(document)
            item.pop("_id", None)
            documents.append(item)
        return documents

    async def get(self, assignment_id: str) -> Mapping[str, Any] | None:
        document = await (await self._collection()).find_one({"id": assignment_id})
        if document is None:
            return None
        item = dict(document)
        item.pop("_id", None)
        return item

    async def insert(self, payload: Mapping[str, Any]) -> str:
        document = dict(payload)
        await (await self._collection()).insert_one(document)
        return str(document["id"])

    async def insert_many(self, payloads: Sequence[Mapping[str, Any]]) -> int:
        documents = [dict(payload) for payload in payloads]
        if not documents:
            return 0
        await (await self._collection()).insert_many(documents)
        return len(documents)

    async def total_count(self) -> int:
        return await (await self._collection()).count_documents({})

    async def count_status(self, status: str) -> int:
        return await (await self._collection()).count_documents({"status": status})

    async def count_high_priority(self, minimum: int = 4) -> int:
        return await (await self._collection()).count_documents({"priority": {"$gte": minimum}})

    async def average_confidence(self) -> float:
        pipeline = [{"$group": {"_id": None, "avg_confidence": {"$avg": "$decision.confidence_score"}}}]
        result = await (await self._collection()).aggregate(pipeline).to_list(1)
        return float(result[0].get("avg_confidence") or 0.0) if result else 0.0

    async def count_with_violations(self) -> int:
        return await (await self._collection()).count_documents({"decision.violations": {"$exists": True, "$ne": []}})

    async def approve_pending(
        self,
        assignment_ids: Sequence[str],
        user_id: str,
        comments: str | None,
        when: datetime,
    ) -> int:
        result = await (await self._collection()).update_many(
            {"id": {"$in": list(assignment_ids)}, "status": "PENDING"},
            {
                "$set": {
                    "status": "APPROVED",
                    "approved_by": user_id,
                    "approved_at": when,
                    "approval_comments": comments,
                    "last_updated": when,
                }
            },
        )
        return int(result.modified_count)

    async def override_pending(
        self,
        assignment_id: str,
        user_id: str,
        reason: str,
        override_decision: str,
        when: datetime,
    ) -> Mapping[str, Any] | None:
        document = await (await self._collection()).find_one_and_update(
            {"id": assignment_id, "status": "PENDING"},
            {
                "$set": {
                    "status": "OVERRIDDEN",
                    "override_reason": reason,
                    "override_by": user_id,
                    "override_at": when,
                    "override_decision": override_decision,
                    "last_updated": when,
                }
            },
            return_document=True,
        )
        if document is None:
            return None
        item = dict(document)
        item.pop("_id", None)
        return item

    async def list_assigned_trainset_ids(self) -> set[str]:
        cursor = (await self._collection()).find(
            {"status": {"$in": ["PENDING", "APPROVED", "OVERRIDDEN"]}},
            {"trainset_id": 1},
        )
        result: set[str] = set()
        async for document in cursor:
            trainset_id = document.get("trainset_id")
            if trainset_id:
                result.add(str(trainset_id))
        return result

    async def get_by_trainset(self, trainset_id: str) -> Mapping[str, Any] | None:
        return await self.get_assignment_by_query({"trainset_id": trainset_id})

    async def get_assignment_by_query(self, query: Mapping[str, Any]) -> Mapping[str, Any] | None:
        document = await (await self._collection()).find_one(dict(query))
        if document is None:
            return None
        item = dict(document)
        item.pop("_id", None)
        return item
