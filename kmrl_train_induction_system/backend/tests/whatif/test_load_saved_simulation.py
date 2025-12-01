"""Test loading saved simulation results"""
import pytest
import json
import uuid
from pathlib import Path
from app.api.simulation import get_simulation_result, _ensure_results_is_array


@pytest.mark.asyncio
async def test_load_saved_simulation():
    """Test loading a saved simulation result"""
    # Create a test simulation file
    simulation_id = str(uuid.uuid4())
    test_data = {
        "simulation_id": simulation_id,
        "timestamp": "2024-01-01T00:00:00",
        "baseline": {"num_inducted_trains": 10},
        "scenario": {"num_inducted_trains": 12},
        "deltas": {"num_inducted_trains": 2},
        "explain_log": ["Test log"],
        "results": [
            {"type": "baseline", "kpis": {}},
            {"type": "scenario", "kpis": {}}
        ]
    }
    
    from app.services.whatif_simulator import SIMULATION_RUNS_DIR
    test_file = SIMULATION_RUNS_DIR / f"{simulation_id}.json"
    
    try:
        # Save test file
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load it
        result = await get_simulation_result(simulation_id)
        
        assert result["simulation_id"] == simulation_id
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 2
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


@pytest.mark.asyncio
async def test_load_simulation_with_object_results():
    """Test loading simulation that has object instead of array (should be fixed)"""
    simulation_id = str(uuid.uuid4())
    test_data = {
        "simulation_id": simulation_id,
        "results": {"type": "baseline"}  # Wrong format - object instead of array
    }
    
    from app.services.whatif_simulator import SIMULATION_RUNS_DIR
    test_file = SIMULATION_RUNS_DIR / f"{simulation_id}.json"
    
    try:
        # Save test file with wrong format
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load it - should be fixed to array
        result = await get_simulation_result(simulation_id)
        
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 1
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


@pytest.mark.asyncio
async def test_load_nonexistent_simulation():
    """Test loading a non-existent simulation returns 404"""
    fake_id = str(uuid.uuid4())
    
    with pytest.raises(Exception):  # Should raise HTTPException
        await get_simulation_result(fake_id)







