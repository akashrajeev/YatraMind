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
        if isinstance(job_cards, dict):
            job_card_id = job_cards.get("job_card_id")
            if not job_card_id:
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
            else:
                # The canonical bulk mapper historically omitted job_card_id.
                # Preserve it here so a real work-order update remains valid.
                snapshot["job_cards"] = {
                    **job_cards,
                    "job_card_id": job_card_id,
                }

        await self._process_bulk_trainset_data_canonical(snapshot)

    async def _process_bulk_trainset_data_canonical(self, data: Dict[str, Any]):
        """Apply the canonical factor mapping while preserving N8N job-card IDs."""
        trainset_id = data.get("trainset_id")
        if not trainset_id:
            return

        if "fitness_certificates" in data:
            certs = data["fitness_certificates"]
            if isinstance(certs, dict):
                for cert_type, cert_data in certs.items():
                    if isinstance(cert_data, dict):
                        await self._update_fitness_factor({
                            "trainset_id": trainset_id,
                            "certificate": cert_type,
                            "status": cert_data.get("status"),
                            "expiry_date": cert_data.get("expiry_date"),
                        })

        if "job_cards" in data and isinstance(data["job_cards"], dict):
            jc = data["job_cards"]
            if jc.get("job_card_id"):
                await self._update_job_card_factor({
                    "trainset_id": trainset_id,
                    "job_card_id": jc.get("job_card_id"),
                    "open_cards": jc.get("open_cards"),
                    "critical_cards": jc.get("critical_cards"),
                })

        if "branding" in data and isinstance(data["branding"], dict):
            branding = data["branding"]
            await self._update_branding_factor({
                "trainset_id": trainset_id,
                "current_advertiser": branding.get("current_advertiser"),
                "priority": branding.get("priority"),
                "revenue_per_day": branding.get("revenue_per_day"),
            })

        if "cleaning" in data and isinstance(data["cleaning"], dict):
            await self._update_cleaning_factor({
                "trainset_id": trainset_id,
                **data["cleaning"],
            })

    @staticmethod
    async def _get_collection_compat(name: str):
        result = cloud_db_manager.get_collection(name)
        if inspect.isawaitable(result):
            result = await result
        return result


__all__ = ["N8NDataIngestionService"]
