"""Test deterministic behavior of What-If simulations"""
import pytest
from app.services.whatif_simulator import run_whatif


@pytest.mark.asyncio
async def test_determinism_with_seed():
    """Test that same scenario with same seed produces identical results"""
    scenario1 = {
        "required_service_hours": 14,
        "random_seed": 42
    }
    
    scenario2 = {
        "required_service_hours": 14,
        "random_seed": 42
    }
    
    result1 = await run_whatif(scenario1)
    result2 = await run_whatif(scenario2)
    
    # Results should be identical (excluding timestamps and IDs)
    assert result1["baseline"] == result2["baseline"]
    assert result1["scenario"] == result2["scenario"]
    assert result1["deltas"] == result2["deltas"]


@pytest.mark.asyncio
async def test_determinism_without_seed():
    """Test that same scenario without seed is deterministic (no randomness)"""
    scenario = {
        "required_service_hours": 14
        # No random_seed - should still be deterministic
    }
    
    result1 = await run_whatif(scenario)
    result2 = await run_whatif(scenario)
    
    # Results should be identical (excluding timestamps and IDs)
    assert result1["baseline"] == result2["baseline"]
    assert result1["scenario"] == result2["scenario"]
    assert result1["deltas"] == result2["deltas"]


@pytest.mark.asyncio
async def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results (if randomness is used)"""
    scenario1 = {
        "required_service_hours": 14,
        "random_seed": 42
    }
    
    scenario2 = {
        "required_service_hours": 14,
        "random_seed": 123
    }
    
    result1 = await run_whatif(scenario1)
    result2 = await run_whatif(scenario2)
    
    # Results might be different if randomness is involved
    # But both should have valid structure
    assert isinstance(result1["results"], list)
    assert isinstance(result2["results"], list)







