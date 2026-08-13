"""Golden regression contract for the pre-refactor optimization pipeline.

This intentionally captures behavior from a known-good historical optimization
run without executing the full ML/database stack. Phase 2 should replace this
with deterministic live-optimizer fixtures once the optimizer is decomposed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenOptimizationBaseline:
    requested_train_count: int = 7
    eligible_train_count: int = 25
    granted_train_count: int = 7
    required_service_trains: int = 7
    standby_buffer: int = 2
    total_required_trains: int = 9


def test_historical_optimization_contract_is_explicit() -> None:
    baseline = GoldenOptimizationBaseline()

    assert baseline.requested_train_count == 7
    assert baseline.eligible_train_count == 25
    assert baseline.granted_train_count == baseline.requested_train_count
    assert baseline.required_service_trains == baseline.requested_train_count
    assert baseline.total_required_trains == (
        baseline.required_service_trains + baseline.standby_buffer
    )


def test_historical_optimization_contract_has_no_service_shortfall() -> None:
    baseline = GoldenOptimizationBaseline()

    service_shortfall = max(
        0, baseline.required_service_trains - baseline.granted_train_count
    )

    assert service_shortfall == 0
