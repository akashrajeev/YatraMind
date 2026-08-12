import pytest

from app.ml.risk_provider import HeuristicRiskProvider
from app.models.trainset import OptimizationRequest
from app.services.optimization_engine import CanonicalOptimizationEngine


def valid_trainset(trainset_id: str, *, branding: bool = False, critical: int = 0):
    return {
        "trainset_id": trainset_id,
        "status": "STANDBY",
        "current_mileage": 1000,
        "max_mileage_before_maintenance": 50000,
        "fitness_certificates": {
            "rolling_stock": {"status": "VALID"},
            "signalling": {"status": "VALID"},
            "telecom": {"status": "VALID"},
        },
        "job_cards": {"open_cards": critical, "critical_cards": critical},
        "branding": {
            "current_advertiser": "Acme" if branding else None,
            "priority": "HIGH" if branding else "LOW",
        },
    }


@pytest.mark.asyncio
async def test_canonical_engine_meets_requested_service_count():
    engine = CanonicalOptimizationEngine(risk_provider=HeuristicRiskProvider())
    decisions, fleet = await engine.optimize(
        [valid_trainset("T-001"), valid_trainset("T-002")],
        OptimizationRequest(required_service_count=1),
    )

    assert fleet.required_service_trains == 1
    assert sum(d.decision == "INDUCT" for d in decisions) == 1
    assert len(decisions) == 2


@pytest.mark.asyncio
async def test_canonical_engine_never_inducts_critical_trainset():
    engine = CanonicalOptimizationEngine(risk_provider=HeuristicRiskProvider())
    decisions, _ = await engine.optimize(
        [valid_trainset("T-001", critical=1), valid_trainset("T-002")],
        OptimizationRequest(required_service_count=1),
    )

    by_id = {decision.trainset_id: decision for decision in decisions}
    assert by_id["T-001"].decision == "MAINTENANCE"
    assert "CRITICAL_JOB_CARD" in by_id["T-001"].violations
    assert by_id["T-002"].decision == "INDUCT"


@pytest.mark.asyncio
async def test_branding_is_ranked_before_tier_three_heuristics():
    engine = CanonicalOptimizationEngine(risk_provider=HeuristicRiskProvider())
    decisions, _ = await engine.optimize(
        [valid_trainset("T-001", branding=False), valid_trainset("T-002", branding=True)],
        OptimizationRequest(required_service_count=1),
    )

    inducted = next(item for item in decisions if item.decision == "INDUCT")
    assert inducted.trainset_id == "T-002"
