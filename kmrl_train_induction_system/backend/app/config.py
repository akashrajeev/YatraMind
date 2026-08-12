# backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from typing import Optional
import os
import yaml
from pathlib import Path

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mongodb_url: str = ""
    database_name: str = "kmrl_db"
    influxdb_url: str = ""
    influxdb_token: str = ""
    influxdb_org: str = ""
    influxdb_bucket: str = "kmrl_sensor_data"
    redis_url: str = ""

    mqtt_broker: str = ""
    mqtt_broker_host: Optional[str] = None
    mqtt_broker_port: Optional[str] = None
    mqtt_use_tls: Optional[str] = None
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    environment: str = "development"
    debug: bool = False
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    gemini_api_key: Optional[str] = None
    model_path: str = "models/"
    confidence_threshold: float = 0.8

    maximo_base_url: Optional[str] = None
    maximo_api_key: Optional[str] = None
    maximo_username: Optional[str] = None
    maximo_password: Optional[str] = None
    drools_service_url: Optional[str] = None
    n8n_webhook_url: Optional[str] = None

    default_hours_per_train: float = 2.0
    max_hours_warning_threshold_multiplier: int = 24
    dev_mock_seed: int = 0
    ml_deterministic_seed: int = 42
    warn_on_unknown_depot: bool = True
    warn_on_capacity_exceeded: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=('settings_',),
    )

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
if os.path.exists("use"):
    load_dotenv("use", override=True)

_defaults_path = Path(__file__).parent / "config" / "defaults.yaml"
_defaults = {}
if _defaults_path.exists():
    try:
        with open(_defaults_path, "r") as f:
            _defaults = yaml.safe_load(f) or {}
    except Exception as e:
        import logging
        logging.warning(f"Could not load defaults.yaml: {e}")

settings = Settings()

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
