import pytest

from app.ml.risk_provider import HeuristicRiskProvider


@pytest.mark.asyncio
async def test_heuristic_risk_provider_is_deterministic():
    provider = HeuristicRiskProvider()
    trainsets = [
        {
            "trainset_id": "T-001",
            "current_mileage": 10000,
            "max_mileage_before_maintenance": 50000,
            "job_cards": {"critical_cards": 0},
        },
        {
            "trainset_id": "T-002",
            "current_mileage": 40000,
            "max_mileage_before_maintenance": 50000,
            "job_cards": {"critical_cards": 1},
        },
    ]
    first = await provider.predict(trainsets)
    second = await provider.predict(trainsets)
    assert first == second
    assert first[0].health_score > first[1].health_score
    assert first[1].risk_probability > first[0].risk_probability
