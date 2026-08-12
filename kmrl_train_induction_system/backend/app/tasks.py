from __future__ import annotations

import asyncio
import logging

from app.celery_app import celery_app
from app.ml.trainer import TrainConfig, train_and_register
from app.services.data_ingestion import DataIngestionService
from app.services.nightly_optimization import NightlyOptimizationService

logger = logging.getLogger(__name__)


@celery_app.task(name="optimization.nightly_run")
def nightly_run_optimization() -> dict:
    async def _run() -> dict:
        try:
            return await NightlyOptimizationService().run()
        except Exception as exc:
            logger.exception("Nightly optimization failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    return asyncio.get_event_loop().run_until_complete(_run())


@celery_app.task(name="ingestion.refresh_all")
def ingestion_refresh_all() -> dict:
    async def _run() -> dict:
        try:
            return await DataIngestionService().ingest_all_sources()
        except Exception as exc:
            logger.exception("Ingestion refresh failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    return asyncio.get_event_loop().run_until_complete(_run())


@celery_app.task(name="ml.train_model")
def train_model() -> dict:
    async def _run() -> dict:
        try:
            return await train_and_register(TrainConfig())
        except Exception as exc:
            logger.exception("Model training failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    return asyncio.get_event_loop().run_until_complete(_run())
