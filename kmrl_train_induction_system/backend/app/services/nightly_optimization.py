"""Nightly optimization application service.

Keeps Celery concerns out of the optimization workflow and delegates data
access to repository adapters.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict
from app.models.trainset import OptimizationRequest
from app.repositories.mongo import MongoTrainsetRepository
from app.repositories.mongo_optimization import MongoOptimizationRepository
from app.services.optimization_service import OptimizationService


class NightlyOptimizationService:
    def __init__(self, trainsets=None, history=None, optimization=None):
        self.trainsets = trainsets or MongoTrainsetRepository()
        self.history = history or MongoOptimizationRepository()
        self.optimization = optimization or OptimizationService()

    async def run(self) -> Dict[str, Any]:
        trainsets = [dict(item) for item in await self.trainsets.list_all()]
        if not trainsets:
            return {"status": "no_data"}

        request = OptimizationRequest(target_date=datetime.utcnow())
        result = await self.optimization.optimize(trainsets, request)
        decisions = result.decisions
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_date": request.target_date.isoformat(),
            "required_service_count": request.required_service_count,
            "service_date": request.service_date,
            "total_decisions": len(decisions),
            "inducted_count": result.inducted_count,
            "standby_count": sum(d.decision == "STANDBY" for d in decisions),
            "maintenance_count": sum(d.decision == "MAINTENANCE" for d in decisions),
            "average_confidence": sum(d.confidence_score for d in decisions) / len(decisions) if decisions else 0,
            "decisions": [d.dict() for d in decisions],
            "fleet_requirement": result.fleet_requirement.dict(),
        }
        await self.history.save_run(payload)
        return {"status": "ok", "decisions": len(decisions)}
