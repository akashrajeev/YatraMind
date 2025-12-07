import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import os
import sys

# Mock modules before importing app
sys.modules["apscheduler.schedulers.asyncio"] = MagicMock()
sys.modules["prometheus_fastapi_instrumentator"] = MagicMock()

from app.main import app
from app.utils.cloud_database import cloud_db_manager

# Mock DB manager
cloud_db_manager.connect_mongodb = AsyncMock()
cloud_db_manager.connect_influxdb = AsyncMock()
cloud_db_manager.get_collection = AsyncMock()
cloud_db_manager.close_all = AsyncMock()

# Mock collection behavior
mock_collection = AsyncMock()
mock_collection.find = MagicMock(return_value=AsyncMock())
mock_collection.find.return_value.__aiter__.return_value = [
    {"trainset_id": "T-001", "status": "STANDBY", "fitness_certificates": {"rolling_stock": {"status": "VALID"}}},
    {"trainset_id": "T-002", "status": "STANDBY", "fitness_certificates": {"rolling_stock": {"status": "VALID"}}}
]
mock_collection.find_one = AsyncMock(return_value={"trainset_id": "T-001", "status": "STANDBY"})
cloud_db_manager.get_collection.return_value = mock_collection

API_KEY = os.getenv("API_KEY", "kmrl_api_key_2024")

@pytest.fixture
def client():
    # Mock scheduler in app.main if it was already imported
    with patch("app.main.scheduler", MagicMock()), \
         patch("app.main.Instrumentator", MagicMock()), \
         patch("app.main.cloud_db_manager", cloud_db_manager):
        with TestClient(app) as c:
            c.headers.update({"X-API-Key": API_KEY})
            yield c

def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_optimization_workflow(client):
    """Test complete optimization workflow"""
    # Test constraints (protected)
    response = client.get("/api/optimization/constraints/check")
    assert response.status_code == 200
    
    # Test optimization (protected)
    request = {
        "target_date": datetime.now().isoformat(),
        "service_date": datetime.now().strftime("%Y-%m-%d"),
        "required_service_hours": 14 # Deprecated but should still work (ignored if service_date used, or used as fallback)
    }
    
    response = client.post("/api/optimization/run", json=request)
    if response.status_code != 200:
        print(response.json())
        
    assert response.status_code == 200
    data = response.json()
    assert "granted_train_count" in data
    assert "decisions" in data
    assert "required_service_trains" in data # New field

def test_task_management(client):
    """Test background tasks"""
    # Trigger task (unprotected)
    with patch("app.celery_app.celery_app.send_task") as mock_send:
        mock_send.return_value = MagicMock(id="test_task_id")
        
        response = client.post("/tasks/optimization/run")
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_id"] == "test_task_id"
    
    # Check task status
    with patch("app.celery_app.celery_app.AsyncResult") as mock_result:
        mock_result.return_value.state = "SUCCESS"
        mock_result.return_value.ready.return_value = True
        mock_result.return_value.result = {"status": "ok"}
        
        resp = client.get("/tasks/status/test_task_id")
        assert resp.status_code == 200
        assert resp.json()["state"] == "SUCCESS"