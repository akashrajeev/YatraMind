"""Compatibility adapter for n8n webhook payloads.

Keeps n8n snapshot payloads separate from true work-order updates and makes the
Celery refresh path tolerant of lightweight test doubles.
"""
from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Dict

from app.services.data_ingestion import DataIngestionService
from app.utils.cloud_database import cloud_db_manager


class N8NDataIngestionService(DataIngestionService):
    """n8n-specific adapter over the canonical ingestion service."""

    async def _trigger_optimization_refresh(self, source: str):
        """Refresh cached optimization state and queue the nightly task."""
        try:
            collection = cloud_db_manager.get_collection("optimization_results")
            if inspect.isawaitable(collection):
                collection = await collection
            delete_many = getattr(collection, "delete_many", None)
            if delete_many is not None:
                result = delete_many({})
                if inspect.isawaitable(result):
                    await result

            from app.celery_app import celery_app
            celery_app.send_task("optimization.nightly_run")
        except Exception as exc:
            # Refresh is intentionally best-effort; ingestion itself must still
            # succeed when Redis/Celery/database infrastructure is unavailable.
            self._log_refresh_failure(source, exc)

    @staticmethod
    def _log_refresh_failure(source: str, exc: Exception) -> None:
        import logging
        logging.getLogger(__name__).error(
            "Failed to trigger optimization refresh for %s: %s", source, exc
        )

    async def _process_bulk_trainset_data(self, data: Dict[str, Any]):
        """Process an n8n trainset snapshot without inventing work-order IDs."""
        if not isinstance(data, dict) or not data.get("trainset_id"):
            return

        snapshot = dict(data)
        job_cards = snapshot.get("job_cards")
        if isinstance(job_cards, dict) and not job_cards.get("job_card_id"):
            # A snapshot such as {critical_cards, open_cards} is fleet state,
            # not a Maximo work-order. Persist it on the trainset and remove it
            # from the generic helper so that helper does not require a fake ID.
            trainsets = await self._get_collection_compat("trainsets")
            update = {
                "job_cards": job_cards,
                "last_updated_sources.job_cards": datetime.now().isoformat(),
            }
            result = trainsets.update_one(
                {"trainset_id": snapshot["trainset_id"]},
                {"$set": update},
                upsert=True,
            )
            if inspect.isawaitable(result):
                await result
            snapshot.pop("job_cards", None)

        await super()._process_bulk_trainset_data(snapshot)

    @staticmethod
    async def _get_collection_compat(name: str):
        result = cloud_db_manager.get_collection(name)
        if inspect.isawaitable(result):
            result = await result
        return result


__all__ = ["N8NDataIngestionService"]
