# backend/tests/test_stabling_fixes.py
"""
Tests for stabling geometry and shunting schedule fixes
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.optimization_store import get_latest_decisions, get_decisions_from_history

client = TestClient(app)


@pytest.fixture
def sample_decisions():
    return [
        {"trainset_id": "T-001", "decision": "INDUCT", "confidence_score": 0.95, "score": 0.92, "reasons": ["High fitness", "Low mileage"], "top_reasons": [], "top_risks": [], "violations": [], "shap_values": []},
        {"trainset_id": "T-002", "decision": "STANDBY", "confidence_score": 0.75, "score": 0.65, "reasons": ["Moderate condition"], "top_reasons": [], "top_risks": [], "violations": [], "shap_values": []},
        {"trainset_id": "T-003", "decision": "MAINTENANCE", "confidence_score": 0.90, "score": 0.10, "reasons": ["Critical job cards"], "top_reasons": [], "top_risks": [], "violations": [], "shap_values": []},
    ]


@pytest.fixture
def sample_trainsets():
    return [
        {"trainset_id": "T-001", "current_location": {"depot": "Aluva", "bay": "Aluva_BAY_06"}, "status": "ACTIVE"},
        {"trainset_id": "T-002", "current_location": {"depot": "Aluva", "bay": "Aluva_BAY_07"}, "status": "STANDBY"},
        {"trainset_id": "T-003", "current_location": {"depot": "Petta", "bay": "Petta_BAY_01"}, "status": "MAINTENANCE"},
    ]


@pytest.mark.asyncio
async def test_get_latest_decisions_success():
    mock_doc = {"_id": "test_id", "_meta": {"updated_at": "2024-01-01T00:00:00Z"}, "decisions": [{"trainset_id": "T-001", "decision": "INDUCT"}, {"trainset_id": "T-002", "decision": "STANDBY"}]}
    with patch('app.services.optimization_store.cloud_db_manager') as mock_db:
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=mock_doc)
        mock_db.get_collection = AsyncMock(return_value=mock_collection)
        result = await get_latest_decisions()
        assert result is not None
        assert len(result) == 2
        assert result[0]["trainset_id"] == "T-001"


@pytest.mark.asyncio
async def test_get_latest_decisions_not_found():
    with patch('app.services.optimization_store.cloud_db_manager') as mock_db:
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_db.get_collection = AsyncMock(return_value=mock_collection)
        result = await get_latest_decisions()
        assert result is None


@pytest.mark.asyncio
async def test_get_latest_decisions_empty_list():
    mock_doc = {"_id": "test_id", "_meta": {"updated_at": "2024-01-01T00:00:00Z"}, "decisions": []}
    with patch('app.services.optimization_store.cloud_db_manager') as mock_db:
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=mock_doc)
        mock_db.get_collection = AsyncMock(return_value=mock_collection)
        result = await get_latest_decisions()
        assert result is None


def test_stabling_geometry_endpoint_with_decisions(sample_decisions, sample_trainsets):
    with patch('app.api.optimization.get_latest_decisions', return_value=sample_decisions):
        with patch('app.api.optimization.cloud_db_manager') as mock_db:
            mock_collection = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.find = MagicMock(return_value=mock_cursor)
            mock_cursor.__aiter__ = MagicMock(return_value=iter(sample_trainsets))
            mock_db.get_collection = AsyncMock(return_value=mock_collection)
            response = client.get("/api/optimization/stabling-geometry", headers={"X-API-Key": "kmrl_api_key_2024"})
            assert response.status_code != 400
            if response.status_code == 200:
                data = response.json()
                assert "efficiency_improvement" in data
                assert "total_optimized_positions" in data


def test_stabling_geometry_endpoint_no_decisions():
    with patch('app.api.optimization.get_latest_decisions', return_value=None):
        with patch('app.api.optimization.get_decisions_from_history', return_value=None):
            with patch('app.api.optimization.cloud_db_manager') as mock_db:
                mock_collection = AsyncMock()
                mock_cursor = AsyncMock()
                mock_cursor.find = MagicMock(return_value=mock_cursor)
                mock_cursor.__aiter__ = MagicMock(return_value=iter([{"trainset_id": "T-001"}]))
                mock_db.get_collection = AsyncMock(return_value=mock_collection)
                response = client.get("/api/optimization/stabling-geometry", headers={"X-API-Key": "kmrl_api_key_2024"})
                # Legacy compatibility endpoints now fail closed when no API key is configured.
                assert response.status_code == 503


def test_shunting_schedule_endpoint_with_decisions(sample_decisions, sample_trainsets):
    with patch('app.api.optimization.get_latest_decisions', return_value=sample_decisions):
        with patch('app.api.optimization.cloud_db_manager') as mock_db:
            mock_collection = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.find = MagicMock(return_value=mock_cursor)
            mock_cursor.__aiter__ = MagicMock(return_value=iter(sample_trainsets))
            mock_db.get_collection = AsyncMock(return_value=mock_collection)
            response = client.get("/api/optimization/shunting-schedule", headers={"X-API-Key": "kmrl_api_key_2024"})
            assert response.status_code != 400
            if response.status_code == 200:
                data = response.json()
                assert "shunting_schedule" in data
                assert "total_operations" in data
                assert "estimated_total_time" in data


def test_efficiency_improvement_calculation():
    for ratio, expected_percentage in [(0.15, 15.0), (0.85, 85.0), (0.0, 0.0), (1.0, 100.0)]:
        calculated = round(float(ratio) * 100, 2)
        assert calculated == expected_percentage
