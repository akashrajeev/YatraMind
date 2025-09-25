# backend/app/celery_app.py
from __future__ import annotations

import os
from celery import Celery
from app.config import settings


def _mongodb_transport_url() -> str:
    # Use MongoDB as Celery broker to avoid Redis/RabbitMQ.
    # We point broker to the same MongoDB cluster but use a separate database.
    # Celery's Kombu supports mongodb transport via 'mongodb://'.
    return settings.mongodb_url


celery_app = Celery(
    "kmrl_backend",
    broker=_mongodb_transport_url(),
    backend=settings.mongodb_url,
)

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


