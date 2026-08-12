from unittest.mock import AsyncMock

import pytest

from app.models.trainset import InductionDecision, OptimizationRequest
from app.services.optimization_service import OptimizationService
from app.services.fleet_planning import FleetRequirementResult


class StubOptimizer:
    def __init__(self, decisions):
        self.decisions = decisions
        self.optimize_calls = []

    async def optimize(self, trainsets, request):
        self.optimize_calls.append((trainsets, request))
        return self.decisions, FleetRequirementResult(
            required_service_trains=1,
            standby_buffer=1,
            total_required_trains=2,
            calculation_method="test",
        )


@pytest.mark.asyncio
async def test_optimization_service_orchestrates_optimizer_and_stabling():
    decisions = [
        InductionDecision(trainset_id="T-001", decision="INDUCT", score=1.0),
        InductionDecision(trainset_id="T-002", decision="STANDBY", score=0.5),
    ]
    optimizer = StubOptimizer(decisions)
    stabling = AsyncMock()
    stabling.optimize_stabling_geometry.return_value = {"optimized_layout": {}}

    service = OptimizationService(optimizer=optimizer, stabling_optimizer=stabling)
    request = OptimizationRequest(required_service_count=1)
    result = await service.optimize([{"trainset_id": "T-001"}], request)

    assert result.inducted_count == 1
    assert result.eligible_count == 1
    assert result.fleet_requirement.total_required_trains == 2
    stabling.optimize_stabling_geometry.assert_awaited_once()
