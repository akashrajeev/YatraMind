"""Test KPI computation and delta calculation"""
import pytest
from app.services.whatif_simulator import _compute_kpis, _generate_explain_log


def test_compute_kpis_structure():
    """Test that compute_kpis returns expected structure"""
    decisions = [
        {"trainset_id": "T-001", "decision": "INDUCT"},
        {"trainset_id": "T-002", "decision": "STANDBY"},
        {"trainset_id": "T-003", "decision": "MAINTENANCE"},
    ]
    
    stabling_geometry = {
        "total_shunting_time": 120,
        "total_turnout_time": 60,
        "optimized_layout": {
            "Aluva": {
                "shunting_operations": [
                    {"trainset_id": "T-001", "estimated_time": 30},
                    {"trainset_id": "T-002", "estimated_time": 20},
                ]
            }
        },
        "efficiency_metrics": {
            "overall_efficiency": 0.85
        },
        "unassigned": []
    }
    
    kpis = _compute_kpis(decisions, stabling_geometry)
    
    assert "num_inducted_trains" in kpis
    assert "num_standby_trains" in kpis
    assert "num_maintenance_trains" in kpis
    assert "total_shunting_time" in kpis
    assert "total_turnout_time" in kpis
    assert "num_shunt_ops" in kpis
    assert "num_unassigned" in kpis
    assert "efficiency_improvement" in kpis
    
    assert kpis["num_inducted_trains"] == 1
    assert kpis["num_standby_trains"] == 1
    assert kpis["num_maintenance_trains"] == 1
    assert kpis["total_shunting_time"] == 120
    assert kpis["num_shunt_ops"] == 2
    assert isinstance(kpis["efficiency_improvement"], (int, float))


def test_generate_explain_log():
    """Test explain log generation"""
    baseline_kpis = {
        "num_inducted_trains": 10,
        "total_shunting_time": 100,
        "num_shunt_ops": 5,
        "efficiency_improvement": 75.0
    }
    
    scenario_kpis = {
        "num_inducted_trains": 12,
        "total_shunting_time": 80,
        "num_shunt_ops": 4,
        "efficiency_improvement": 80.0
    }
    
    scenario = {
        "required_service_hours": 16
    }
    
    explain_log = _generate_explain_log(baseline_kpis, scenario_kpis, scenario)
    
    assert isinstance(explain_log, list)
    assert len(explain_log) > 0
    assert all(isinstance(msg, str) for msg in explain_log)
    
    # Should mention the changes
    log_text = " ".join(explain_log)
    assert "inducted" in log_text.lower() or "shunting" in log_text.lower()


@pytest.mark.asyncio
async def test_deltas_computed_correctly():
    """Test that deltas are computed correctly in full simulation"""
    from app.services.whatif_simulator import run_whatif
    
    scenario = {
        "required_service_hours": 20  # Different from baseline
    }
    
    result = await run_whatif(scenario)
    
    # Check deltas structure
    deltas = result["deltas"]
    assert "num_inducted_trains" in deltas
    assert "total_shunting_time" in deltas
    assert "efficiency_improvement" in deltas
    
    # Deltas should be scenario - baseline
    baseline = result["baseline"]
    scenario_kpis = result["scenario"]
    
    assert deltas["num_inducted_trains"] == scenario_kpis["num_inducted_trains"] - baseline["num_inducted_trains"]
    assert deltas["total_shunting_time"] == scenario_kpis["total_shunting_time"] - baseline["total_shunting_time"]







