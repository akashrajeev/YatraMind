#!/usr/bin/env python3
"""
Development startup script for KMRL Train Induction System
Sets required environment variables and starts the FastAPI server
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent

def set_default(key: str, value: str) -> None:
    """Only set env var if it's not already defined (e.g. via .env)."""
    if os.environ.get(key) is None:
        os.environ[key] = value

# Load .env (so values defined there win)
load_dotenv(BASE_DIR / ".env")

# Set environment variables for development (fallbacks)
set_default("API_KEY", "kmrl_api_key_2024")
set_default("SECRET_KEY", "kmrl-secret-key-2024")
set_default("DEBUG", "true")
set_default("ENVIRONMENT", "development")

# Set database URLs to use mock data unless overridden
set_default("MONGODB_URL", "mongodb://localhost:27017/kmrl_db")
set_default("DATABASE_NAME", "kmrl_db")

# Set InfluxDB to use mock data
set_default("INFLUXDB_URL", "https://us-west-2-1.aws.cloud2.influxdata.com")
set_default("INFLUXDB_TOKEN", "mock_token")
set_default("INFLUXDB_ORG", "mock_org")
set_default("INFLUXDB_BUCKET", "kmrl_sensor_data")

# Set Redis to use local instance by default
set_default("REDIS_URL", "redis://localhost:6379")

# Set MQTT configuration defaults
set_default("MQTT_BROKER", "broker.hivemq.com")
set_default("MQTT_PORT", "1883")

# Set ML model configuration defaults
set_default("MODEL_PATH", "models/")
set_default("CONFIDENCE_THRESHOLD", "0.8")

print("🚀 Starting KMRL Train Induction System in Development Mode")
print("=" * 60)
print(f"API Key: {os.environ['API_KEY']}")
print(f"Debug Mode: {os.environ['DEBUG']}")
print(f"Environment: {os.environ['ENVIRONMENT']}")
print("=" * 60)

# Change to the backend directory
os.chdir(BASE_DIR)

# Start the FastAPI server
try:
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "app.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload",
        "--log-level", "info"
    ], check=True)
except KeyboardInterrupt:
    print("\n👋 Shutting down KMRL Train Induction System")
except subprocess.CalledProcessError as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)
