import pytest

from app.services.dashboard_service import DashboardService


class FakeDashboardRepository:
    async def list_trainsets(self):
        return [
            {
                "trainset_id": "T-001",
                "status": "ACTIVE",
                "fitness_certificates": {"rolling": {"status": "VALID"}},
                "job_cards": {"open_cards": 2, "critical_cards": 0},
                "current_location": {"depot": "Aluva"},
            },
            {
                "trainset_id": "T-002",
                "status": "MAINTENANCE",
                "fitness_certificates": {"rolling": {"status": "EXPIRED"}},
                "job_cards": {"open_cards": 3, "critical_cards": 1},
                "current_location": {"depot": "Muttom"},
            },
        ]

    async def get_latest_induction(self):
        return {"decisions": [{"trainset_id": "T-001", "decision": "INDUCT"}, {"trainset_id": "T-002", "decision": "MAINTENANCE"}]}

    async def count_pending_assignments(self):
        return 1

    async def list_assigned_trainset_ids(self):
        return {"T-001"}

    async def get_recent_optimization_history(self, limit=7):
        return [{"average_confidence": 0.8}, {"average_confidence": 0.9}]

    async def list_alert_candidates(self):
        return [
            {
                "trainset_id": "T-002",
                "fitness_certificates": {"rolling": {"status": "EXPIRED"}},
                "job_cards": {"critical_cards": 1},
                "current_mileage": 49000,
                "max_mileage_before_maintenance": 50000,
            }
        ]


@pytest.mark.asyncio
async def test_dashboard_overview_uses_repository_data():
    result = await DashboardService(FakeDashboardRepository()).overview()
    assert result["total_trainsets"] == 2
    assert result["fleet_status"] == {"active": 1, "maintenance": 1, "standby": 0}
    assert result["fitness_certificates"]["expired"] == 1
    assert result["job_cards"]["critical"] == 1
    assert result["pending_assignments"] == 2


@pytest.mark.asyncio
async def test_dashboard_alerts_are_sorted_by_severity():
    result = await DashboardService(FakeDashboardRepository()).alerts()
    assert result["total_alerts"] == 3
    assert result["alerts"][0]["type"] == "CRITICAL"
    assert result["alerts"][1]["type"] == "HIGH"


@pytest.mark.asyncio
async def test_dashboard_performance_uses_recent_history():
    result = await DashboardService(FakeDashboardRepository()).performance()
    assert result["optimization_performance"]["total_runs"] == 2
    assert result["optimization_performance"]["average_confidence_score"] == 0.85
