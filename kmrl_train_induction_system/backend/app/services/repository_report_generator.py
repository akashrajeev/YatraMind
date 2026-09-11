"""Repository-backed report generator adapter."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from app.repositories.mongo_reports import MongoReportRepository
from app.services.report_generator import ReportGenerator


class RepositoryReportGenerator(ReportGenerator):
    """Keep report rendering reusable while sourcing report data through a repository."""

    def __init__(self, repository: MongoReportRepository | None = None):
        self.repository = repository or MongoReportRepository()
        super().__init__()

    async def _get_fleet_overview_data(self, target_date: datetime) -> Dict[str, Any]:
        return dict(await self.repository.fleet_overview())

    async def _get_assignment_summary_data(self, target_date: datetime) -> Dict[str, Any]:
        return dict(await self.repository.assignment_summary())

    async def _get_critical_alerts_data(self, target_date: datetime) -> List[Dict[str, Any]]:
        return [dict(item) for item in await self.repository.critical_alerts()]
