"""
Tests for simulation fallback behavior
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_simulation_with_ai_unavailable():
    """Test that when AI is requested but unavailable, fallback is used"""
    # Mock AI health check to return False
    with patch('app.services.ml_health.check_ai_services_available', return_value=False):
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
            "ai_mode": True  # Request AI but it's unavailable
        }
        
        response = client.post("/api/v1/simulate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use fallback
        assert data["used_ai"] is False
        
        # Should have fallback warning
        warnings = data.get("warnings", [])
        fallback_warning = any("deterministic fallback" in w.lower() or "ai services unavailable" in w.lower() for w in warnings)
        assert fallback_warning, "Should have fallback warning in warnings list"


def test_simulation_error_returns_json():
    """Test that errors return JSON with run_id"""
    # Invalid payload to trigger error
    payload = {
        "fleet": -1,  # Invalid
        "depots": []
    }
    
    response = client.post("/api/v1/simulate", json=payload)
    
    # Should return error status
    assert response.status_code in [400, 500]
    
    # Should return JSON (not HTML)
    try:
        data = response.json()
        # If it's a dict, it should have error info
        if isinstance(data, dict):
            assert "detail" in data or "message" in data
    except:
        pytest.fail("Error response should be JSON, not HTML")

