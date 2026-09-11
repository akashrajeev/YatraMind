"""Application service for append-only audit records."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from app.models.audit import AuditLog, AuditLogCreate
from app.repositories.mongo_audit import MongoAuditRepository


class AuditService:
    """Coordinate audit persistence while keeping API code storage-agnostic."""

    def __init__(self, repository: MongoAuditRepository | None = None) -> None:
        self.repository = repository or MongoAuditRepository()

    async def record(self, audit_log: AuditLogCreate) -> str:
        return await self.repository.create(audit_log.dict())

    async def list_logs(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        user_id: str | None = None,
        action: str | None = None,
        limit: int = 10000,
    ) -> list[AuditLog]:
        documents = await self.repository.list(
            start=start,
            end=end,
            user_id=user_id,
            action=action,
            limit=limit,
        )
        result: list[AuditLog] = []
        for document in documents:
            try:
                result.append(AuditLog(**dict(document)))
            except Exception:
                continue
        return result


audit_service = AuditService()
