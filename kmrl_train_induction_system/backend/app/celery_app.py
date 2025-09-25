# backend/app/celery_app.py
from __future__ import annotations

import os
from celery import Celery
from app.config import settings


def _broker_url() -> str:
    # Celery + Kombu mongodb transport does not support mongodb+srv on Windows reliably.
    # For local/dev on Windows, fall back to in-memory broker.
    url = settings.mongodb_url
    if isinstance(url, str) and url.startswith("mongodb+srv://"):
        return "memory://"
    return url


celery_app = Celery("kmrl_backend", broker=_broker_url(), backend=settings.mongodb_url)

# Transport/backend options for MongoDB
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


