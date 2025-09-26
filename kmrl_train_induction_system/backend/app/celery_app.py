# backend/app/celery_app.py
from __future__ import annotations

import os
from celery import Celery
from app.config import settings


def _is_srv_mongo(url: str | None) -> bool:
    return isinstance(url, str) and url.startswith("mongodb+srv://")


def _broker_url() -> str:
    """Return Celery broker URL.

    Redis is intentionally not used. Prefer in-memory broker for local/dev,
    otherwise fall back to MongoDB if not using mongodb+srv.
    """
    # Always use in-memory broker for local/dev scenarios to avoid Redis
    if _is_srv_mongo(settings.mongodb_url):
        return "memory://"
    # If not using SRV, still avoid Redis and prefer Mongo as broker
    return settings.mongodb_url


def _result_backend_url() -> str:
    """Return Celery result backend URL.

    Avoid Redis entirely. Use SQLite when mongodb+srv is present; otherwise use MongoDB.
    """
    if _is_srv_mongo(settings.mongodb_url):
        return "db+sqlite:///celery_results.sqlite3"
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


