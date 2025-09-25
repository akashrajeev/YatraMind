# backend/app/tasks.py
from __future__ import annotations

from typing import List
from datetime import datetime
import logging

from app.celery_app import celery_app
from app.utils.cloud_database import cloud_db_manager
from app.services.optimizer import TrainInductionOptimizer
from app.models.trainset import OptimizationRequest, InductionDecision
from app.api.optimization import store_optimization_history, write_optimization_metrics
from app.services.data_ingestion import DataIngestionService
from app.ml.trainer import train_and_register, TrainConfig
from app.ml.predictor import batch_predict

logger = logging.getLogger(__name__)


@celery_app.task(name="optimization.nightly_run")
def nightly_run_optimization() -> dict:
    """Background task to run nightly optimization and persist results."""
    import asyncio

    async def _run() -> dict:
        try:
            # Load trainsets
            collection = await cloud_db_manager.get_collection("trainsets")
            cursor = collection.find({})
            trainsets_data = []
            async for doc in cursor:
                doc.pop("_id", None)
                trainsets_data.append(doc)
            if not trainsets_data:
                return {"status": "no_data"}

            # Predict risk and persist to MongoDB
            features_for_pred = [{"trainset_id": t.get("trainset_id"), **{k: v for k, v in t.items() if isinstance(v, (int, float))}} for t in trainsets_data]
            preds = await batch_predict(features_for_pred)
            risk_map = {p["trainset_id"]: p for p in preds}
            # Update trainsets collection with risk
            ts_col = await cloud_db_manager.get_collection("trainsets")
            for t in trainsets_data:
                rid = t.get("trainset_id")
                pr = risk_map.get(rid, {})
                if pr:
                    await ts_col.update_one({"trainset_id": rid}, {"$set": {"predicted_failure_risk": pr.get("risk_prob", 0.2), "risk_top_features": pr.get("top_features", [])}}, upsert=False)

            optimizer = TrainInductionOptimizer()
            req = OptimizationRequest(target_date=datetime.now(), required_service_hours=14)
            result: List[InductionDecision] = await optimizer.optimize(trainsets_data, req)

            # Persist
            await store_optimization_history(req, result)
            await write_optimization_metrics(result)
            return {"status": "ok", "decisions": len(result)}
        except Exception as e:
            logger.exception(f"Nightly optimization failed: {e}")
            return {"status": "error", "error": str(e)}

    return asyncio.get_event_loop().run_until_complete(_run())


@celery_app.task(name="ingestion.refresh_all")
def ingestion_refresh_all() -> dict:
    """Pull snapshots from Maximo/Sheets and write UNS."""
    import asyncio

    async def _run() -> dict:
        svc = DataIngestionService()
        res = await svc.ingest_all_sources()
        return res

    return asyncio.get_event_loop().run_until_complete(_run())


@celery_app.task(name="ml.train_model")
def train_model() -> dict:
    """Trigger ML model retraining."""
    import asyncio
    async def _run():
        try:
            res = await train_and_register(TrainConfig())
            return res
        except Exception as e:
            logger.exception(f"Model training failed: {e}")
            return {"status": "error", "error": str(e)}
    return asyncio.get_event_loop().run_until_complete(_run())


