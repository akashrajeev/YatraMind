# backend/app/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MongoDB Atlas Configuration
    mongodb_url: str = "mongodb+srv://username:password@cluster.mongodb.net/kmrl_db?retryWrites=true&w=majority"
    database_name: str = "kmrl_db"
    
    # InfluxDB Cloud Configuration
    influxdb_url: str = "https://us-west-2-1.aws.cloud2.influxdata.com"
    influxdb_token: str = "your_influxdb_token_here"
    influxdb_org: str = "your_org_id"
    influxdb_bucket: str = "kmrl_sensor_data"
    
    # Redis Cloud Configuration
    redis_url: str = "redis://username:password@redis-12345.c1.us-west-2-1.ec2.cloud.redislabs.com:12345"
    
    # MQTT Configuration
    mqtt_broker: str = "broker.hivemq.com"
    mqtt_broker_host: Optional[str] = None
    mqtt_broker_port: Optional[str] = None
    mqtt_use_tls: Optional[str] = None
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    
    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    environment: Optional[str] = None
    debug: bool = True
    
    # ML Model Configuration
    model_path: str = "models/"
    confidence_threshold: float = 0.8
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

# Create settings instance
settings = Settings()
