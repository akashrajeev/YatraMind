"""Mongo-backed audit log repository."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.repositories.protocols import AuditRepository
from app.utils.cloud_database import cloud_db_manager


class MongoAuditRepository(AuditRepository):
    """Persist audit records without leaking Mongo access into route modules."""

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
