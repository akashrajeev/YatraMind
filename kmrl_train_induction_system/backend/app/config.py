# backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MongoDB Atlas Configuration
    mongodb_url: str = Field(
        default="mongodb+srv://username:password@cluster.mongodb.net/kmrl_db?retryWrites=true&w=majority",
        env="MONGODB_URL",
    )
    database_name: str = Field(default="kmrl_db", env="DATABASE_NAME")
    
    # InfluxDB Cloud Configuration
    influxdb_url: str = Field(default="https://us-west-2-1.aws.cloud2.influxdata.com", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="your_influxdb_token_here", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="your_org_id", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="kmrl_sensor_data", env="INFLUXDB_BUCKET")
    
    # Redis Cloud Configuration
    redis_url: str = Field(default="redis://username:password@redis-12345.c1.us-west-2-1.ec2.cloud.redislabs.com:12345", env="REDIS_URL")
    
    # MQTT Configuration
    mqtt_broker: str = Field(default="broker.hivemq.com", env="MQTT_BROKER")
    mqtt_broker_host: Optional[str] = Field(default=None, env="MQTT_BROKER_HOST")
    mqtt_broker_port: Optional[str] = Field(default=None, env="MQTT_BROKER_PORT")
    mqtt_use_tls: Optional[str] = Field(default=None, env="MQTT_USE_TLS")
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    
    # API Configuration
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    secret_key: Optional[str] = Field(default=None, env="SECRET_KEY")
    environment: Optional[str] = Field(default=None, env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # ML Model Configuration
    model_path: str = Field(default="models/", env="MODEL_PATH")
    confidence_threshold: float = Field(default=0.8, env="CONFIDENCE_THRESHOLD")
    
    # Maximo API Configuration
    maximo_base_url: Optional[str] = Field(default=None, env="MAXIMO_BASE_URL")
    maximo_api_key: Optional[str] = Field(default=None, env="MAXIMO_API_KEY")
    maximo_username: Optional[str] = Field(default=None, env="MAXIMO_USERNAME")
    maximo_password: Optional[str] = Field(default=None, env="MAXIMO_PASSWORD")
    
    # Optional external Drools service
    drools_service_url: Optional[str] = Field(default=None, env="DROOLS_SERVICE_URL")

    # N8N Configuration
    n8n_webhook_url: Optional[str] = Field(default=None, env="N8N_WEBHOOK_URL")
    
    # pydantic-settings v2 style configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=('settings_',),  # Fix warning for model_path field
    )

    # No custom __init__; rely on pydantic-settings to read from .env and env vars

# Eagerly load .env and fallback 'use' file so cloud creds are picked up reliably
load_dotenv(".env")
if os.path.exists("use"):
    load_dotenv("use", override=True)

# Create settings instance
settings = Settings()
