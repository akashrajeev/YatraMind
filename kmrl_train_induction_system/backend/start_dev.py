#!/usr/bin/env python3
"""
Development startup script for KMRL Train Induction System
Sets required environment variables and starts the FastAPI server
"""

import os
import sys
import subprocess
from pathlib import Path

# Set environment variables for development
os.environ["API_KEY"] = "kmrl_api_key_2024"
os.environ["SECRET_KEY"] = "kmrl-secret-key-2024"
os.environ["DEBUG"] = "true"
os.environ["ENVIRONMENT"] = "development"

# Set database URLs to use mock data
os.environ["MONGODB_URL"] = "mongodb://localhost:27017/kmrl_db"
os.environ["DATABASE_NAME"] = "kmrl_db"

# Set InfluxDB to use mock data
os.environ["INFLUXDB_URL"] = "https://us-west-2-1.aws.cloud2.influxdata.com"
os.environ["INFLUXDB_TOKEN"] = "mock_token"
os.environ["INFLUXDB_ORG"] = "mock_org"
os.environ["INFLUXDB_BUCKET"] = "kmrl_sensor_data"

# Set Redis to use local instance
os.environ["REDIS_URL"] = "redis://localhost:6379"

# Set MQTT configuration
os.environ["MQTT_BROKER"] = "broker.hivemq.com"
os.environ["MQTT_PORT"] = "1883"

# Set ML model configuration
os.environ["MODEL_PATH"] = "models/"
os.environ["CONFIDENCE_THRESHOLD"] = "0.8"

print("🚀 Starting KMRL Train Induction System in Development Mode")
print("=" * 60)
print(f"API Key: {os.environ['API_KEY']}")
print(f"Debug Mode: {os.environ['DEBUG']}")
print(f"Environment: {os.environ['ENVIRONMENT']}")
print("=" * 60)

# Change to the backend directory
backend_dir = Path(__file__).parent
os.chdir(backend_dir)

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
