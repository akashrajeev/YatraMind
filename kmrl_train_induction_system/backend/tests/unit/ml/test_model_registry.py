import asyncio

import pytest

from app.ml.metrics import InferenceMetrics
from app.ml.risk_provider import HeuristicRiskProvider, RiskPrediction


def test_inference_metrics_track_calls_failures_and_latency():
    metrics = InferenceMetrics()
    metrics.record(0.20)
    metrics.record(0.10, failed=True)

    assert metrics.calls == 2
    assert metrics.failures == 1
    assert metrics.average_latency_seconds == pytest.approx(0.15)


def test_heuristic_provider_sets_stable_model_metadata():
    provider = HeuristicRiskProvider()
    result = asyncio.run(
        provider.predict([{
            "trainset_id": "T-001",
            "current_mileage": 10000,
            "max_mileage_before_maintenance": 50000,
            "job_cards": {"critical_cards": 0},
        }])
    )

    assert isinstance(result[0], RiskPrediction)
    assert result[0].provider == "heuristic-v1"
    assert result[0].model_version == "heuristic-v1"
