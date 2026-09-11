import pytest

from app.services.dashboard_service import DashboardService


class FakeDashboardRepository:
    async def list_trainsets(self):
        return [
            {
                "trainset_id": "T-001",
                "status": "ACTIVE",
                "fitness_certificates": {"A": {"status": "VALID"}},
                "job_cards": {"open_cards": 2, "critical_cards": 0},
                "current_location": {"depot": "ALV"},
                "current_mileage": 10000,
                "max_mileage_before_maintenance": 50000,
            },
            {
                "trainset_id": "T-002",
                "status": "MAINTENANCE",
                "fitness_certificates": {"A": {"status": "EXPIRED"}},
                "job_cards": {"open_cards": 3, "critical_cards": 1},
                "current_location": {"depot": "MUT"},
                "current_mileage": 49000,
                "max_mileage_before_maintenance": 50000,
            },
        ]

    async def get_latest_induction(self):
        return {
            "decisions": [
                {"trainset_id": "T-001", "decision": "INDUCT"},
                {"trainset_id": "T-002", "decision": "MAINTENANCE"},
            ]
        }

    async def count_pending_assignments(self):
        return 1

    async def list_assigned_trainset_ids(self):
        return {"T-001"}

    async def list_alert_candidates(self):
        return await self.list_trainsets()

    async def get_recent_optimization_history(self, limit=7):
        return [{"average_confidence": 0.9}, {"average_confidence": 0.8}]


@pytest.mark.asyncio
async def test_dashboard_overview_uses_repository_data():
    service = DashboardService(FakeDashboardRepository())
    result = await service.overview()

    assert result["total_trainsets"] == 2
    assert result["fleet_status"] == {"active": 1, "maintenance": 1, "standby": 0}
    assert result["fitness_certificates"]["expired"] == 1
    assert result["job_cards"]["critical"] == 1
    assert result["depot_distribution"] == {"ALV": 1, "MUT": 1}


@pytest.mark.asyncio
async def test_dashboard_performance_averages_recent_history():
    service = DashboardService(FakeDashboardRepository())
    result = await service.performance()

    assert result["optimization_performance"]["total_runs"] == 2
    assert result["optimization_performance"]["average_confidence_score"] == 0.85
