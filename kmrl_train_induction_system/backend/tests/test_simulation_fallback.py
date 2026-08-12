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
    with patch('app.services.ml_health.check_ai_services_available', return_value=False):
        payload = {
            "fleet": 5,
            "depots": [{
                "name": "Muttom",
                "location_type": "FULL_DEPOT",
                "service_bays": 6,
                "maintenance_bays": 4,
                "standby_bays": 2
            }],
            "ai_mode": True
        }
        response = client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["used_ai"] is False
        warnings = data.get("warnings", [])
        assert any("deterministic fallback" in w.lower() or "ai services unavailable" in w.lower() for w in warnings)


def test_simulation_error_returns_json():
    """Invalid Pydantic input is rejected with a JSON 422 response."""
    payload = {"fleet": -1, "depots": []}
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert isinstance(data, dict)
    assert "detail" in data
