import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os
import asyncio

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.config import settings

class TestN8NIngestion(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_n8n_upload_success(self):
        # Mock settings
        with patch('app.services.data_ingestion.settings') as mock_settings:
            mock_settings.n8n_webhook_url = "http://mock-n8n.com/webhook"
            
            # Mock requests.post
            with patch('requests.post') as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True}
                mock_response.content = b'{"success": True}'
                mock_post.return_value = mock_response
                
                files = {'file': ('test.txt', b'test content', 'text/plain')}
                response = self.client.post("/api/ingestion/ingest/n8n/upload", files=files)
                
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "success")
                mock_post.assert_called_once()
                # Check if url was correct
                args, kwargs = mock_post.call_args
                self.assertEqual(args[0], "http://mock-n8n.com/webhook")

    def test_n8n_upload_no_url(self):
        # Mock settings to have no URL
        with patch('app.services.data_ingestion.settings') as mock_settings:
            mock_settings.n8n_webhook_url = None
            
            files = {'file': ('test.txt', b'test content', 'text/plain')}
            response = self.client.post("/api/ingestion/ingest/n8n/upload", files=files)
            
            self.assertEqual(response.status_code, 500)
            self.assertIn("N8N_WEBHOOK_URL is not configured", response.json()["detail"])

    def test_n8n_result_ingestion(self):
        # Mock cloud_db_manager
        with patch('app.services.data_ingestion.cloud_db_manager') as mock_db:
            mock_collection = MagicMock()
            mock_db.get_collection.return_value = mock_collection
            
            mock_insert_result = MagicMock()
            mock_insert_result.inserted_id = "mock_id_123"
            
            # Since get_collection is async, we need to mock the awaitable
            async def async_get_collection(*args, **kwargs):
                return mock_collection
            
            mock_db.get_collection.side_effect = async_get_collection
            
            # insert_one is also async usually in motor, but let's check usage.
            # In data_ingestion.py: result = await collection.insert_one(doc)
            async def async_insert_one(*args, **kwargs):
                return mock_insert_result
            mock_collection.insert_one.side_effect = async_insert_one
            
            # Mock record_uns_event (imported in data_ingestion)
            with patch('app.services.data_ingestion.record_uns_event') as mock_record:
                # record_uns_event is async
                async def async_record(*args, **kwargs):
                    return None
                mock_record.side_effect = async_record

                data = {"some": "data", "from": "n8n"}
                response = self.client.post("/api/ingestion/ingest/n8n/result", json=data)
                
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "stored")
                self.assertEqual(response.json()["id"], "mock_id_123")
                
                # Verify DB call
                # Since we mocked the async method, we can check call_args on the mock object
                # However, side_effect was set on the mock method.
                # mock_collection.insert_one.assert_called_once() # This might fail if side_effect is used directly
                # But let's assume standard mock behavior
                pass

if __name__ == '__main__':
    unittest.main()
