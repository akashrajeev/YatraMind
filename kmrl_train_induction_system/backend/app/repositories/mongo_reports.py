"""Mongo-backed report read repository."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.repositories.protocols import ReportRepository
from app.utils.cloud_database import cloud_db_manager


class MongoReportRepository(ReportRepository):
    """Provide report-oriented read models over Mongo collections."""

    async def fleet_overview(self) -> Mapping[str, Any]:
        cursor = (await cloud_db_manager.get_collection("trainsets")).find({})
        total = active = maintenance = standby = valid = expired = 0
        async for document in cursor:
            total += 1
            status = str(document.get("status", "STANDBY")).upper()
            if status == "ACTIVE":
                active += 1
            elif status == "MAINTENANCE":
                maintenance += 1
            else:
                standby += 1
            for certificate in (document.get("fitness_certificates") or {}).values():
                certificate_status = str(certificate.get("status", "")).upper()
                if certificate_status == "VALID":
                    valid += 1
                elif certificate_status == "EXPIRED":
                    expired += 1
        return {
            "Total Trainsets": total,
            "Active": active,
            "Maintenance": maintenance,
            "Standby": standby,
            "Valid Certificates": valid,
            "Expired Certificates": expired,
        }

    async def assignment_summary(self) -> Mapping[str, Any]:
        collection = await cloud_db_manager.get_collection("assignments")
        return {
            "Total Assignments": await collection.count_documents({}),
            "Pending": await collection.count_documents({"status": "PENDING"}),
            "Approved": await collection.count_documents({"status": "APPROVED"}),
            "Overridden": await collection.count_documents({"status": "OVERRIDDEN"}),
        }

    async def critical_alerts(self, limit: int = 10) -> Sequence[Mapping[str, Any]]:
        cursor = (await cloud_db_manager.get_collection("alerts")).find({"type": "CRITICAL"}).limit(limit)
        result: list[Mapping[str, Any]] = []
        async for document in cursor:
            result.append({
                "Trainset ID": document.get("trainset_id", ""),
                "Message": document.get("message", ""),
                "Timestamp": document.get("timestamp", ""),
                "Category": document.get("category", ""),
            })
        return result
