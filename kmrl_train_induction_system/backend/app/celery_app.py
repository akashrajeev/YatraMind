# backend/app/celery_app.py
from __future__ import annotations

import os
from celery import Celery
from app.config import settings


def _is_srv_mongo(url: str | None) -> bool:
    return isinstance(url, str) and url.startswith("mongodb+srv://")


def _broker_url() -> str:
    # Check for Redis URL from environment (Railway)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis_url
    # On Windows/dev with mongodb+srv, use in-memory broker
    if _is_srv_mongo(settings.mongodb_url):
        return "memory://"
    return settings.mongodb_url


def _result_backend_url() -> str:
    # Celery MongoDB backend doesn't support mongodb+srv reliably.
    # Use SQLite file backend for cross-process result persistence in dev/Windows.
    if _is_srv_mongo(settings.mongodb_url):
        # Store results in local file within backend directory
        return "db+sqlite:///celery_results.sqlite3"
    # Check for Redis URL from environment (Railway)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis_url
    # Otherwise, use MongoDB result backend URL
    return settings.mongodb_url


result_backend_url = _result_backend_url()
celery_app = Celery("kmrl_backend", broker=_broker_url(), backend=result_backend_url)

# Configure transport/backends
if result_backend_url.startswith("db+sqlite"):
    # SQLite result backend configuration
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="Asia/Kolkata",
        enable_utc=True,
    )
else:
    # MongoDB backend configuration
    celery_app.conf.update(
        broker_transport_options={
            "database": "kmrl_celery",
            "collection": "celery_messages",
        },
        result_backend="mongodb",
        result_backend_transport_options={
            "database": "kmrl_celery",
            "taskmeta_collection": "celery_taskmeta",
        },
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="Asia/Kolkata",
        enable_utc=True,
    )


@celery_app.task(name="health.ping")
def ping() -> str:
    return "pong"


