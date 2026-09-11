"""Application service for append-only audit records."""
from __future__ import annotations

from typing import Any, Mapping

from app.models.audit import AuditLogCreate
from app.repositories.mongo_audit import MongoAuditRepository


class AuditService:
    def __init__(self, repository: MongoAuditRepository | None = None) -> None:
        self.repository = repository or MongoAuditRepository()

    async def record(self, audit_log: AuditLogCreate) -> str:
        return await self.repository.create(audit_log.dict())


audit_service = AuditService()
