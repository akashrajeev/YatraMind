"""
Tests for simulation response schema alignment
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_simulation_response_has_required_keys():
    """Test that simulation response contains all required keys in global_summary"""
    payload = {
        "fleet": 5,
        "depots": [
            {
                "name": "Muttom",
                "location_type": "FULL_DEPOT",
                "service_bays": 6,
                "maintenance_bays": 4,
                "standby_bays": 2
            }
        ],
        "ai_mode": False
    }
    
    response = client.post("/api/v1/simulate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check top-level keys
    assert "run_id" in data
    assert "used_ai" in data
    assert "warnings" in data
    assert "global_summary" in data
    
    # Check global_summary has all required keys
    global_summary = data["global_summary"]
    required_keys = [
        "service_trains",
        "required_service",
        "stabled_service",
        "service_shortfall",
        "shunting_time",
        "turnout_time",
        "total_capacity",
        "fleet",
        "transfers_recommended"
    ]
    
    for key in required_keys:
        assert key in global_summary, f"Missing key: {key}"
        assert isinstance(global_summary[key], (int, float)), f"Key {key} should be numeric"
    
    # Verify specific values
    assert global_summary["service_trains"] >= 0
    assert global_summary["required_service"] >= 0
    assert global_summary["shunting_time"] >= 0
    assert global_summary["fleet"] == 5


def test_simulation_fallback_sets_used_ai_false():
    """Test that fallback path sets used_ai=false"""
    payload = {
        "fleet": 5,
        "depots": [
            {
                "name": "Muttom",
                "location_type": "FULL_DEPOT",
                "service_bays": 6,
                "maintenance_bays": 4,
                "standby_bays": 2
            }
        ],
        "ai_mode": False
    }
    
    response = client.post("/api/v1/simulate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["used_ai"] is False


def test_simulation_deterministic_with_seed():
    """Test that deterministic fallback is reproducible with same seed"""
    payload = {
        "fleet": 10,
        "depots": [
            {
                "name": "Muttom",
                "location_type": "FULL_DEPOT",
                "service_bays": 6,
                "maintenance_bays": 4,
                "standby_bays": 2
            }
        ],
        "ai_mode": False,
        "seed": 12345
    }
    
    # Run twice with same seed
    response1 = client.post("/api/v1/simulate", json=payload)
    response2 = client.post("/api/v1/simulate", json=payload)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    # Results should be identical with same seed
    assert data1["global_summary"]["service_trains"] == data2["global_summary"]["service_trains"]
    assert data1["global_summary"]["shunting_time"] == data2["global_summary"]["shunting_time"]

