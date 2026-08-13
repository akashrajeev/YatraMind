import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


class TestN8NIngestion(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('app.celery_app.celery_app')
    def test_n8n_triggers_optimization(self, mock_celery):
        with patch('app.services.data_ingestion.cloud_db_manager') as mock_db:
            mock_collection = MagicMock()
            async def async_get_collection(*args, **kwargs): return mock_collection
            mock_db.get_collection.side_effect = async_get_collection
            async def async_insert_one(*args, **kwargs): return MagicMock(inserted_id="trigger_test")
            mock_collection.insert_one.side_effect = async_insert_one
            async def async_update_one(*args, **kwargs): return None
            mock_collection.update_one.side_effect = async_update_one
            with patch('app.services.data_ingestion.record_uns_event', new_callable=AsyncMock):
                data = {"updates": [{"source_type": "fitness", "data": {"trainset_id": "T-001", "certificate": "Test"}}]}
                response = self.client.post("/api/v1/ingestion/ingest/n8n/result?apply_updates=true", json=data)
                self.assertEqual(response.status_code, 200)
                mock_celery.send_task.assert_called_with("optimization.nightly_run")

    def test_n8n_upload_success(self):
        with patch('app.services.data_ingestion.settings') as mock_settings:
            mock_settings.n8n_webhook_url = "http://mock-n8n.com/webhook"
            with patch('httpx.AsyncClient') as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__aenter__.return_value = mock_client
                mock_response = MagicMock(status_code=200)
                mock_response.json.return_value = {"success": True}
                mock_response.content = b'{"success": true}'
                async def async_post(*args, **kwargs): return mock_response
                mock_client.post.side_effect = async_post
                files = [
                    ('files', ('test1.txt', b'content1', 'text/plain')),
                    ('files', ('test2.json', b'{"a": 1}', 'application/json')),
                ]
                response = self.client.post("/api/v1/ingestion/ingest/n8n/upload", files=files)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "success")
                self.assertIn("2 file(s)", response.json()["message"])
                mock_client.post.assert_called_once()
                _, kwargs = mock_client.post.call_args
                self.assertEqual(len(kwargs['files']), 2)

    def test_n8n_upload_no_url(self):
        with patch('app.services.data_ingestion.settings') as mock_settings:
            mock_settings.n8n_webhook_url = None
            response = self.client.post("/api/v1/ingestion/ingest/n8n/upload", files={'file': ('test.txt', b'test content', 'text/plain')})
            self.assertEqual(response.status_code, 500)
            self.assertIn("N8N_WEBHOOK_URL is not configured", response.json()["detail"])

    def test_n8n_result_ingestion(self):
        with patch('app.services.data_ingestion.cloud_db_manager') as mock_db:
            mock_collection = MagicMock()
            mock_db.get_collection = AsyncMock(return_value=mock_collection)
            mock_insert_result = MagicMock(inserted_id="mock_id_123")
            mock_collection.insert_one = AsyncMock(return_value=mock_insert_result)
            mock_collection.update_one = AsyncMock(return_value=None)
            with patch('app.services.data_ingestion.record_uns_event', new_callable=AsyncMock):
                response = self.client.post("/api/v1/ingestion/ingest/n8n/result", json={"some": "data", "from": "n8n"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "stored_and_processed")
                self.assertEqual(response.json()["id"], "mock_id_123")

    @patch('app.celery_app.celery_app')
    def test_n8n_raw_list_payload(self, mock_celery):
        with patch('app.services.data_ingestion.cloud_db_manager') as mock_db:
            mock_collection = MagicMock()
            mock_db.get_collection.side_effect = AsyncMock(return_value=mock_collection)
            mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="raw_list_id"))
            mock_collection.update_one = AsyncMock(return_value=None)
            with patch('app.services.data_ingestion.record_uns_event', new_callable=AsyncMock):
                data = [
                    {"trainset_id": "T-009", "status": "ERR_MAINT", "job_cards": {"job_card_id": "JC-009", "critical_cards": 2, "open_cards": 5}},
                    {"trainset_id": "T-010", "status": "READY"},
                ]
                response = self.client.post("/api/v1/ingestion/ingest/n8n/result?apply_updates=true", json=data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["updates_processed"], 2)
                mock_celery.send_task.assert_called_with("optimization.nightly_run")

    def test_n8n_result_switch_false(self):
        with patch('app.services.data_ingestion.cloud_db_manager') as mock_db:
            mock_collection = MagicMock()
            mock_db.get_collection.side_effect = AsyncMock(return_value=mock_collection)
            mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="mock_id_switch"))
            mock_collection.update_one = AsyncMock(return_value=None)
            with patch('app.services.data_ingestion.record_uns_event', new_callable=AsyncMock):
                data = {"updates": [{"source_type": "fitness", "data": {"trainset_id": "T-001", "certificate": "Test"}}]}
                response = self.client.post("/api/v1/ingestion/ingest/n8n/result?apply_updates=true", json=data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["updates_processed"], 1)
                self.assertIn("stored_and_processed", response.json()["status"])

    def test_n8n_router_logic(self):
        with patch('app.services.data_ingestion.cloud_db_manager') as mock_db:
            mock_collection = MagicMock()
            mock_db.get_collection.side_effect = AsyncMock(return_value=mock_collection)
            mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="mock_id_router"))
            mock_collection.update_one = AsyncMock(return_value=None)
            with patch('app.services.data_ingestion.record_uns_event', new_callable=AsyncMock):
                with patch('app.services.data_ingestion.DataIngestionService._update_fitness_factor', new_callable=AsyncMock) as mock_fitness, \
                     patch('app.services.data_ingestion.DataIngestionService._update_job_card_factor', new_callable=AsyncMock) as mock_job, \
                     patch('app.services.data_ingestion.DataIngestionService._update_branding_factor', new_callable=AsyncMock) as mock_branding:
                    data = {"updates": [
                        {"source_type": "fitness", "data": {"trainset_id": "T-001", "certificate": "Test Cert"}},
                        {"source_type": "job_card", "data": {"job_card_id": "WO-1", "trainset_id": "T-001"}},
                        {"source_type": "branding", "data": {"trainset_id": "T-001", "current_advertiser": "Ad"}},
                    ]}
                    response = self.client.post("/api/v1/ingestion/ingest/n8n/result", json=data)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["updates_processed"], 3)
                    mock_fitness.assert_awaited_once()
                    mock_job.assert_awaited_once()
                    mock_branding.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
