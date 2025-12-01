"""Test basic What-If simulation functionality"""
import pytest
from app.services.whatif_simulator import run_whatif


@pytest.mark.asyncio
async def test_run_whatif_simple_scenario():
    """Test running a simple What-If scenario"""
    scenario = {
        "required_service_hours": 14
    }
    
    result = await run_whatif(scenario)
    
    # Check required fields
    assert "simulation_id" in result
    assert "timestamp" in result
    assert "baseline" in result
    assert "scenario" in result
    assert "deltas" in result
    assert "explain_log" in result
    assert "results" in result
    
    # CRITICAL: results must be an array
    assert isinstance(result["results"], list), "results must be an array"
    assert len(result["results"]) == 2, "results must contain baseline and scenario"
    
    # Check results structure
    baseline_result = result["results"][0]
    scenario_result = result["results"][1]
    
    assert baseline_result["type"] == "baseline"
    assert scenario_result["type"] == "scenario"
    assert "kpis" in baseline_result
    assert "kpis" in scenario_result


@pytest.mark.asyncio
async def test_run_whatif_no_crashes():
    """Test that What-If simulation doesn't crash with various scenarios"""
    scenarios = [
        {"required_service_hours": 10},
        {"required_service_hours": 20},
        {"force_decisions": {"T-001": "INDUCT"}},
        {"override_train_attributes": {"T-001": {"current_mileage": 10000}}},
    ]
    
    for scenario in scenarios:
        try:
            result = await run_whatif(scenario)
            assert isinstance(result["results"], list)
        except Exception as e:
            pytest.fail(f"What-If simulation crashed with scenario {scenario}: {e}")







