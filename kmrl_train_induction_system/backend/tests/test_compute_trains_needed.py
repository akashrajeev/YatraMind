"""Unit tests for compute_trains_needed function (hours to trains conversion)"""
import pytest
from app.services.optimizer import compute_trains_needed


def test_compute_trains_needed_with_estimated_hours():
    """Test conversion when trainsets have estimated_service_hours"""
    trainsets = [
        {"trainset_id": "T-001", "estimated_service_hours": 2.0},
        {"trainset_id": "T-002", "estimated_service_hours": 2.5},
        {"trainset_id": "T-003", "estimated_service_hours": 1.5},
    ]
    
    # 14 hours needed, avg 2.0 hours/train = 7 trains needed
    result = compute_trains_needed(14.0, trainsets)
    assert result == 7
    assert result <= len(trainsets)


def test_compute_trains_needed_without_estimated_hours():
    """Test conversion using default hours per train"""
    trainsets = [
        {"trainset_id": "T-001"},
        {"trainset_id": "T-002"},
        {"trainset_id": "T-003"},
    ]
    
    # 14 hours needed, default 2.0 hours/train = 7 trains needed
    result = compute_trains_needed(14.0, trainsets)
    assert result == 7
    assert result <= len(trainsets)


def test_compute_trains_needed_clamps_to_available():
    """Test that result doesn't exceed available trains"""
    trainsets = [{"trainset_id": f"T-{i:03d}"} for i in range(1, 6)]  # 5 trains
    
    # Request 100 hours (would need 50 trains), but only 5 available
    result = compute_trains_needed(100.0, trainsets)
    assert result == 5
    assert result <= len(trainsets)


def test_compute_trains_needed_always_returns_at_least_one():
    """Test that result is always at least 1"""
    trainsets = [{"trainset_id": "T-001"}]
    
    result = compute_trains_needed(0.1, trainsets)  # Very small hours
    assert result >= 1


def test_compute_trains_needed_invalid_hours():
    """Test handling of invalid required_hours"""
    trainsets = [{"trainset_id": "T-001"}]
    
    # Negative hours should default to 1 train
    result = compute_trains_needed(-5.0, trainsets)
    assert result == 1
    
    # Zero hours should default to 1 train
    result = compute_trains_needed(0.0, trainsets)
    assert result == 1


def test_compute_trains_needed_empty_trainsets():
    """Test handling of empty trainset list"""
    result = compute_trains_needed(14.0, [])
    assert result == 0


def test_compute_trains_needed_mixed_estimated_hours():
    """Test with mix of trainsets with and without estimated_service_hours"""
    trainsets = [
        {"trainset_id": "T-001", "estimated_service_hours": 2.0},
        {"trainset_id": "T-002"},  # No estimated hours
        {"trainset_id": "T-003", "estimated_service_hours": 3.0},
    ]
    
    # Only T-001 and T-003 have estimated hours, avg = 2.5
    # 14 hours / 2.5 = 5.6 -> ceil = 6 trains
    result = compute_trains_needed(14.0, trainsets)
    assert result == 6
    assert result <= len(trainsets)








