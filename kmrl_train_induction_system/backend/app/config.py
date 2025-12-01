# backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
from typing import Optional
import os
import yaml
from pathlib import Path

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
    
    # Optimization defaults
    default_hours_per_train: float = Field(default=2.0, env="DEFAULT_HOURS_PER_TRAIN")
    max_hours_warning_threshold_multiplier: int = Field(default=24, env="MAX_HOURS_WARNING_THRESHOLD_MULTIPLIER")
    dev_mock_seed: int = Field(default=0, env="DEV_MOCK_SEED")
    ml_deterministic_seed: int = Field(default=42, env="ML_DETERMINISTIC_SEED")
    warn_on_unknown_depot: bool = Field(default=True, env="WARN_ON_UNKNOWN_DEPOT")
    warn_on_capacity_exceeded: bool = Field(default=True, env="WARN_ON_CAPACITY_EXCEEDED")
    
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

# Load defaults from YAML if available
_defaults_path = Path(__file__).parent / "config" / "defaults.yaml"
_defaults = {}
if _defaults_path.exists():
    try:
        with open(_defaults_path, "r") as f:
            _defaults = yaml.safe_load(f) or {}
    except Exception as e:
        import logging
        logging.warning(f"Could not load defaults.yaml: {e}")

# Create settings instance
settings = Settings()

# Override settings with defaults.yaml values if not set in environment
if _defaults:
    if "DEFAULT_HOURS_PER_TRAIN" in _defaults and not os.getenv("DEFAULT_HOURS_PER_TRAIN"):
        settings.default_hours_per_train = float(_defaults["DEFAULT_HOURS_PER_TRAIN"])
    if "MAX_HOURS_WARNING_THRESHOLD_MULTIPLIER" in _defaults and not os.getenv("MAX_HOURS_WARNING_THRESHOLD_MULTIPLIER"):
        settings.max_hours_warning_threshold_multiplier = int(_defaults["MAX_HOURS_WARNING_THRESHOLD_MULTIPLIER"])
    if "DEV_MOCK_SEED" in _defaults and not os.getenv("DEV_MOCK_SEED"):
        settings.dev_mock_seed = int(_defaults["DEV_MOCK_SEED"])
    if "ML_DETERMINISTIC_SEED" in _defaults and not os.getenv("ML_DETERMINISTIC_SEED"):
        settings.ml_deterministic_seed = int(_defaults["ML_DETERMINISTIC_SEED"])
    if "WARN_ON_UNKNOWN_DEPOT" in _defaults and not os.getenv("WARN_ON_UNKNOWN_DEPOT"):
        settings.warn_on_unknown_depot = bool(_defaults["WARN_ON_UNKNOWN_DEPOT"])
    if "WARN_ON_CAPACITY_EXCEEDED" in _defaults and not os.getenv("WARN_ON_CAPACITY_EXCEEDED"):
        settings.warn_on_capacity_exceeded = bool(_defaults["WARN_ON_CAPACITY_EXCEEDED"])

# Simple configuration dict for hours mode & simulation save dir
DEFAULTS = {
    "AVG_HOURS_PER_TRAIN": 12,
    "SIMULATION_SAVE_DIR": "backend/simulation_runs",
    "ALLOW_HOURS_MODE": True,
}


def get_config() -> dict:
    """
    Lightweight config accessor for optimization-related defaults.
    """
    cfg = DEFAULTS.copy()

    avg = getattr(settings, "default_hours_per_train", None)
    if isinstance(avg, (int, float)) and avg > 0:
        cfg["AVG_HOURS_PER_TRAIN"] = float(avg)

    return cfg
