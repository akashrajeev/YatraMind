"""Test path-based override setter"""
import pytest
from app.services.whatif_simulator import _set_nested_value


def test_set_nested_value_simple():
    """Test setting a simple nested value"""
    obj = {}
    _set_nested_value(obj, "fitness.telecom.status", "VALID")
    
    assert obj["fitness"]["telecom"]["status"] == "VALID"


def test_set_nested_value_deep():
    """Test setting a deeply nested value"""
    obj = {}
    _set_nested_value(obj, "a.b.c.d.e", 42)
    
    assert obj["a"]["b"]["c"]["d"]["e"] == 42


def test_set_nested_value_overwrites_existing():
    """Test that setting overwrites existing values"""
    obj = {"fitness": {"telecom": {"status": "EXPIRED"}}}
    _set_nested_value(obj, "fitness.telecom.status", "VALID")
    
    assert obj["fitness"]["telecom"]["status"] == "VALID"


def test_set_nested_value_creates_intermediate():
    """Test that intermediate keys are created if missing"""
    obj = {}
    _set_nested_value(obj, "new.path.value", "test")
    
    assert "new" in obj
    assert "path" in obj["new"]
    assert obj["new"]["path"]["value"] == "test"


@pytest.mark.asyncio
async def test_override_train_attributes_in_simulation():
    """Test that override_train_attributes works in simulation"""
    from app.utils.snapshot import capture_snapshot
    from app.services.whatif_simulator import _apply_overrides
    
    snapshot = await capture_snapshot()
    
    scenario = {
        "override_train_attributes": {
            "T-001": {
                "fitness_certificates.telecom.status": "EXPIRED",
                "current_mileage": 50000
            }
        }
    }
    
    scenario_snapshot = _apply_overrides(snapshot, scenario)
    
    # Find T-001 in scenario snapshot
    trainset = next((t for t in scenario_snapshot["trainsets"] if t.get("trainset_id") == "T-001"), None)
    
    if trainset:
        # Check that overrides were applied
        fitness = trainset.get("fitness_certificates", {})
        telecom = fitness.get("telecom", {})
        assert telecom.get("status") == "EXPIRED"
        assert trainset.get("current_mileage") == 50000







