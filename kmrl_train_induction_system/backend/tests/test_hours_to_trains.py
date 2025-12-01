"""
Unit tests for compute_trains_needed helper to ensure hours → trains conversion works.
"""

from typing import List, Dict, Any

from app.services.optimizer import compute_trains_needed


def _make_trains(estimated_hours: List[float]) -> List[Dict[str, Any]]:
    return [
        {"trainset_id": f"T-{idx+1:03d}", "estimated_service_hours": hours}
        for idx, hours in enumerate(estimated_hours)
    ]


def test_zero_or_negative_hours_defaults_to_one_train():
    trains = _make_trains([10.0, 12.0, 8.0])
    assert compute_trains_needed(0, trains) == 1
    assert compute_trains_needed(-5, trains) == 1


def test_hours_less_than_average_still_requires_one_train():
    trains = _make_trains([10.0, 10.0, 10.0])
    assert compute_trains_needed(4, trains) == 1  # ceil(4/10) -> 1


def test_multiple_of_average_calculates_exact_trains():
    trains = _make_trains([8.0, 8.0, 8.0, 8.0])
    assert compute_trains_needed(16, trains) == 2
    assert compute_trains_needed(24, trains) == 3


def test_requested_hours_clamped_to_available_trains():
    trains = _make_trains([12.0, 12.0])
    assert compute_trains_needed(60, trains) == 2  # would be 5 but clamp to len(trains)
"""
Tests for hours → train-count conversion helper.

We reuse the existing compute_trains_needed helper from the optimizer to
validate the core conversion logic that backs required_service_hours.
"""
from typing import List, Dict, Any

from app.services.optimizer import compute_trains_needed


def _make_trains(estimated_hours: List[float]) -> List[Dict[str, Any]]:
    return [{"trainset_id": f"T-{i+1:03d}", "estimated_service_hours": h} for i, h in enumerate(estimated_hours)]


def test_hours_zero_defaults_to_one_train_when_candidates_exist():
    trains = _make_trains([10.0, 12.0, 8.0])
    # required_hours <= 0 returns at least 1 train when candidates exist
    assert compute_trains_needed(0, trains) == 1
    assert compute_trains_needed(-5, trains) == 1


def test_hours_less_than_average_rounds_up_to_one_train():
    trains = _make_trains([10.0, 10.0, 10.0])  # avg 10 h/train
    # 4 hours still needs 1 train (ceil)
    assert compute_trains_needed(4, trains) == 1


def test_hours_multiple_of_average_computes_exact_train_count():
    trains = _make_trains([8.0, 8.0, 8.0, 8.0])  # avg 8 h/train
    # 16 hours → 2 trains, 24 hours → 3 trains
    assert compute_trains_needed(16, trains) == 2
    assert compute_trains_needed(24, trains) == 3


def test_hours_clamped_to_available_trains():
    trains = _make_trains([12.0, 12.0])  # 2 trains available
    # 60 hours would suggest 5 trains at 12 h/train, but clamp to 2
    assert compute_trains_needed(60, trains) == 2



