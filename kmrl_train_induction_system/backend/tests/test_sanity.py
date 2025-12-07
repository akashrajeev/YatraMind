import pytest
from unittest.mock import MagicMock
import sys

# Mock potential trouble modules
sys.modules["apscheduler.schedulers.asyncio"] = MagicMock()
sys.modules["prometheus_fastapi_instrumentator"] = MagicMock()
sys.modules["app.utils.cloud_database"] = MagicMock()

try:
    from app.main import app
except ImportError as e:
    print(f"ImportError: {e}")
    raise

def test_sanity():
    assert True
