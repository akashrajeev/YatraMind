"""Mongo-backed audit log repository."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.repositories.protocols import AuditRepository
from app.utils.cloud_database import cloud_db_manager


class MongoAuditRepository(AuditRepository):
    """Persist and query audit records without leaking Mongo into API modules."""

    def __init__(self, collection_name: str = "audit_logs") -> None:
        self.collection_name = collection_name

    async def create(self, payload: Mapping[str, Any]) -> str:
        import uuid
        document = dict(payload)
        audit_id = str(document.get("id") or uuid.uuid4())
        document["id"] = audit_id
        document.setdefault("timestamp", datetime.now(timezone.utc))
        await (await cloud_db_manager.get_collection(self.collection_name)).insert_one(document)
        return audit_id

    async def list(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        user_id: str | None = None,
        action: str | None = None,
        limit: int = 10000,
    ) -> Sequence[Mapping[str, Any]]:
        query: dict[str, Any] = {}
        if start or end:
            timestamp: dict[str, datetime] = {}
            if start:
                timestamp["$gte"] = start
            if end:
                timestamp["$lte"] = end
            query["timestamp"] = timestamp
        if user_id:
            query["user_id"] = user_id
        if action:
            query["action"] = action

        cursor = (await cloud_db_manager.get_collection(self.collection_name)).find(query).sort("timestamp", -1).limit(limit)
        result: list[Mapping[str, Any]] = []
        async for document in cursor:
            item = dict(document)
            item.pop("_id", None)
            result.append(item)
        return result
