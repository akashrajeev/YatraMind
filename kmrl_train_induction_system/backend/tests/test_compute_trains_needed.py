"""Unit tests for compute_trains_needed function (hours to trains conversion)."""
from app.services.optimizer import compute_trains_needed


def test_compute_trains_needed_with_estimated_hours_clamps_to_available():
    """Estimated-hour conversion respects the available fleet size."""
    trainsets = [
        {"trainset_id": "T-001", "estimated_service_hours": 2.0},
        {"trainset_id": "T-002", "estimated_service_hours": 2.5},
        {"trainset_id": "T-003", "estimated_service_hours": 1.5},
    ]
    # Average = 2 hours/train, so 14 hours requests 7 trains;
    # the available fleet caps the result at 3.
    result = compute_trains_needed(14.0, trainsets)
    assert result == 3
    assert result <= len(trainsets)


def test_compute_trains_needed_without_estimated_hours_clamps_to_available():
    """Default conversion also respects the available fleet size."""
    trainsets = [
        {"trainset_id": "T-001"},
        {"trainset_id": "T-002"},
        {"trainset_id": "T-003"},
    ]
    result = compute_trains_needed(14.0, trainsets)
    assert result == 3
    assert result <= len(trainsets)


def test_compute_trains_needed_clamps_to_available():
    """Test that result doesn't exceed available trains."""
    trainsets = [{"trainset_id": f"T-{i:03d}"} for i in range(1, 6)]
    result = compute_trains_needed(100.0, trainsets)
    assert result == 5
    assert result <= len(trainsets)


def test_compute_trains_needed_always_returns_at_least_one():
    """Test that result is always at least 1 when trains exist."""
    trainsets = [{"trainset_id": "T-001"}]
    result = compute_trains_needed(0.1, trainsets)
    assert result >= 1


def test_compute_trains_needed_invalid_hours():
    """Test handling of invalid required_hours."""
    trainsets = [{"trainset_id": "T-001"}]
    assert compute_trains_needed(-5.0, trainsets) == 1
    assert compute_trains_needed(0.0, trainsets) == 1


def test_compute_trains_needed_empty_trainsets():
    """Test handling of empty trainset list."""
    assert compute_trains_needed(14.0, []) == 0


def test_compute_trains_needed_mixed_estimated_hours_clamps_to_available():
    """Mixed estimated hours use their average, then clamp to fleet size."""
    trainsets = [
        {"trainset_id": "T-001", "estimated_service_hours": 2.0},
        {"trainset_id": "T-002"},
        {"trainset_id": "T-003", "estimated_service_hours": 3.0},
    ]
    # Average of known values = 2.5 hours/train; 14 hours requests 6,
    # but only 3 trainsets are available.
    result = compute_trains_needed(14.0, trainsets)
    assert result == 3
    assert result <= len(trainsets)
