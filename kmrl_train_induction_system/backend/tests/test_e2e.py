import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import os

from app.main import app
from app.utils.cloud_database import cloud_db_manager
from app.services.auth_service import get_current_user
from app.models.user import User, UserRole

cloud_db_manager.connect_mongodb = AsyncMock()
cloud_db_manager.connect_influxdb = AsyncMock()
cloud_db_manager.get_collection = AsyncMock()
cloud_db_manager.close_all = AsyncMock()

mock_collection = AsyncMock()
mock_collection.find = MagicMock(return_value=AsyncMock())
mock_collection.find.return_value.__aiter__.return_value = [
    {"trainset_id": "T-001", "status": "STANDBY", "fitness_certificates": {"rolling_stock": {"status": "VALID"}}},
    {"trainset_id": "T-002", "status": "STANDBY", "fitness_certificates": {"rolling_stock": {"status": "VALID"}}},
]
mock_collection.find_one = AsyncMock(return_value={"trainset_id": "T-001", "status": "STANDBY"})
cloud_db_manager.get_collection.return_value = mock_collection

ADMIN_USER = User(
    id="test-admin",
    username="test-admin",
    name="Test Admin",
    role=UserRole.ADMIN,
    permissions=[],
    is_active=True,
    is_approved=True,
    email_verified=True,
)

@pytest.fixture
def client():
    app.dependency_overrides[get_current_user] = lambda: ADMIN_USER
    with patch("app.main.scheduler", MagicMock()), patch("app.main.Instrumentator", MagicMock()), patch("app.main.cloud_db_manager", cloud_db_manager):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_optimization_workflow(client):
    request = {
        "target_date": datetime.now().isoformat(),
        "service_date": datetime.now().strftime("%Y-%m-%d"),
        "required_service_hours": 14,
    }
    response = client.post("/api/v1/optimization/run", json=request)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "granted_train_count" in data
    assert "decisions" in data
    assert "required_service_trains" in data


def test_task_management(client):
    with patch("app.celery_app.celery_app.send_task") as mock_send:
        mock_send.return_value = MagicMock(id="test_task_id")
        response = client.post("/tasks/optimization/run")
        assert response.status_code == 200
        assert response.json()["task_id"] == "test_task_id"

    with patch("app.celery_app.celery_app.AsyncResult") as mock_result:
        mock_result.return_value.state = "SUCCESS"
        mock_result.return_value.ready.return_value = True
        mock_result.return_value.result = {"status": "ok"}
        resp = client.get("/tasks/status/test_task_id")
        assert resp.status_code == 200
        assert resp.json()["state"] == "SUCCESS"
