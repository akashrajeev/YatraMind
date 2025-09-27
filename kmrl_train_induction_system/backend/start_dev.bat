@echo off
echo Starting KMRL Train Induction System Backend...
echo.

REM Set environment variables
set API_KEY=kmrl_api_key_2024
set SECRET_KEY=kmrl-secret-key-2024
set DEBUG=true
set ENVIRONMENT=development
set MONGODB_URL=mongodb://localhost:27017/kmrl_db
set DATABASE_NAME=kmrl_db
set INFLUXDB_URL=https://us-west-2-1.aws.cloud2.influxdata.com
set INFLUXDB_TOKEN=mock_token
set INFLUXDB_ORG=mock_org
set INFLUXDB_BUCKET=kmrl_sensor_data
set REDIS_URL=redis://localhost:6379
set MQTT_BROKER=broker.hivemq.com
set MQTT_PORT=1883
set MODEL_PATH=models/
set CONFIDENCE_THRESHOLD=0.8

echo Environment variables set:
echo API_KEY=%API_KEY%
echo DEBUG=%DEBUG%
echo ENVIRONMENT=%ENVIRONMENT%
echo.

REM Start the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

pause
