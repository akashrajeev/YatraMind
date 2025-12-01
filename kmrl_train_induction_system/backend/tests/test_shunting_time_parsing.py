"""Unit tests for shunting time parsing (numeric field)"""
import pytest
from app.services.stabling_optimizer import StablingGeometryOptimizer


@pytest.fixture
def optimizer():
    return StablingGeometryOptimizer()


@pytest.fixture
def sample_operations():
    """Sample shunting operations with numeric estimated_time"""
    return [
        {
            "trainset_id": "T-001",
            "from_bay": 1,
            "to_bay": 6,
            "estimated_time": 8,  # Numeric
            "complexity": "MEDIUM"
        },
        {
            "trainset_id": "T-002",
            "from_bay": 2,
            "to_bay": 7,
            "estimated_time": 12,  # Numeric
            "complexity": "HIGH"
        },
        {
            "trainset_id": "T-003",
            "from_bay": 3,
            "to_bay": 8,
            "estimated_time": 5,  # Numeric
            "complexity": "LOW"
        }
    ]


def test_get_shunting_schedule_uses_numeric_estimated_time(optimizer, sample_operations):
    """Test that get_shunting_schedule uses numeric estimated_time field"""
    optimized_layout = {
        "Aluva": {
            "shunting_operations": sample_operations
        }
    }
    
    import asyncio
    schedule = asyncio.run(optimizer.get_shunting_schedule(optimized_layout))
    
    # Check all entries have numeric estimated_time
    for entry in schedule:
        assert "estimated_time" in entry
        assert isinstance(entry["estimated_time"], int)
        assert entry["estimated_time"] > 0
    
    # Check total time calculation
    total = sum(entry["estimated_time"] for entry in schedule)
    assert total == 25  # 8 + 12 + 5


def test_get_shunting_schedule_handles_non_numeric_estimated_time(optimizer):
    """Test that non-numeric estimated_time is converted or defaulted"""
    optimized_layout = {
        "Aluva": {
            "shunting_operations": [
                {
                    "trainset_id": "T-001",
                    "from_bay": 1,
                    "to_bay": 6,
                    "estimated_time": "8",  # String that can be converted
                    "complexity": "MEDIUM"
                },
                {
                    "trainset_id": "T-002",
                    "from_bay": 2,
                    "to_bay": 7,
                    "estimated_time": None,  # Invalid
                    "complexity": "HIGH"
                }
            ]
        }
    }
    
    import asyncio
    schedule = asyncio.run(optimizer.get_shunting_schedule(optimized_layout))
    
    # T-001 should have converted string to int
    assert schedule[0]["estimated_time"] == 8
    
    # T-002 should have defaulted to 0
    assert schedule[1]["estimated_time"] == 0


def test_get_shunting_schedule_sorts_by_numeric_time(optimizer, sample_operations):
    """Test that schedule is sorted by numeric estimated_time"""
    optimized_layout = {
        "Aluva": {
            "shunting_operations": sample_operations
        }
    }
    
    import asyncio
    schedule = asyncio.run(optimizer.get_shunting_schedule(optimized_layout))
    
    # Should be sorted by estimated_time (5, 8, 12)
    times = [entry["estimated_time"] for entry in schedule]
    assert times == sorted(times)


def test_get_shunting_schedule_includes_both_numeric_and_string_fields(optimizer):
    """Test that both estimated_time (numeric) and estimated_duration (string) are included"""
    optimized_layout = {
        "Aluva": {
            "shunting_operations": [
                {
                    "trainset_id": "T-001",
                    "from_bay": 1,
                    "to_bay": 6,
                    "estimated_time": 8,
                    "complexity": "MEDIUM"
                }
            ]
        }
    }
    
    import asyncio
    schedule = asyncio.run(optimizer.get_shunting_schedule(optimized_layout))
    
    entry = schedule[0]
    assert "estimated_time" in entry
    assert isinstance(entry["estimated_time"], int)
    assert entry["estimated_time"] == 8
    
    assert "estimated_duration" in entry
    assert isinstance(entry["estimated_duration"], str)
    assert entry["estimated_duration"] == "8 minutes"








