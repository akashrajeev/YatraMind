"""Canonical optimization API with legacy-route compatibility.

All existing optimization routes are retained from the legacy router except
POST /run, which now uses the application service and repository boundaries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.config import settings
from app.models.trainset import InductionDecision, OptimizationRequest
from app.models.user import User, UserRole
from app.repositories.mongo import MongoTrainsetRepository
from app.repositories.mongo_optimization import MongoOptimizationRepository
from app.security import require_role
from app.services.optimization_service import OptimizationService
from app.services.optimization_store import get_latest_decisions, get_decisions_from_history
from app.api import optimization_legacy

logger = logging.getLogger(__name__)

router = APIRouter()


def _deterministic_value_from_id(trainset_id: str) -> float:
    """Return a stable pseudo-random value without importing random."""
    digest = hashlib.sha256(str(trainset_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# Preserve all legacy endpoints other than the optimization entrypoint. The
# existing route objects retain their dependencies, response models, and URLs.
for route in optimization_legacy.router.routes:
    if getattr(route, "path", None) != "/run":
        router.routes.append(route)


@router.post("/run")
async def run_optimization(
    background_tasks: BackgroundTasks,
    request: OptimizationRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Run the canonical optimization application service."""
    del current_user
    try:
        trainsets = [dict(item) for item in await MongoTrainsetRepository().list_all()]
        if not trainsets:
            raise HTTPException(status_code=404, detail="No trainsets found")

        result = await OptimizationService().optimize(trainsets, request)
        decisions = result.decisions
        fleet_req = result.fleet_requirement
        granted = result.inducted_count

        diagnostics: Dict[str, Any] = {
            "required_service_trains": fleet_req.required_service_trains,
            "standby_buffer": fleet_req.standby_buffer,
            "calculation_method": fleet_req.calculation_method,
            "eligible_train_count": result.eligible_count,
            "granted_train_count": granted,
            "fleet_requirement": fleet_req.model_dump() if hasattr(fleet_req, "model_dump") else fleet_req.dict(),
        }

        history_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_date": request.target_date.isoformat(),
            "required_service_count": request.required_service_count,
            "service_date": request.service_date,
            "total_decisions": len(decisions),
            "inducted_count": granted,
            "standby_count": sum(d.decision == "STANDBY" for d in decisions),
            "maintenance_count": sum(d.decision == "MAINTENANCE" for d in decisions),
            "average_confidence": sum(d.confidence_score for d in decisions) / len(decisions) if decisions else 0,
            "decisions": [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in decisions],
            "fleet_requirement": fleet_req.model_dump() if hasattr(fleet_req, "model_dump") else fleet_req.dict(),
        }
        await MongoOptimizationRepository().save_run(history_payload)

        if hasattr(optimization_legacy, "write_optimization_metrics"):
            background_tasks.add_task(optimization_legacy.write_optimization_metrics, decisions)

        try:
            sim_dir = Path(getattr(settings, "SIMULATION_SAVE_DIR", "backend/simulation_runs"))
            sim_dir.mkdir(parents=True, exist_ok=True)
            optimization_id = str(uuid.uuid4())
            snapshot = {
                "optimization_id": optimization_id,
                "timestamp": datetime.utcnow().isoformat(),
                "diagnostics": diagnostics,
                "decisions": [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in decisions],
            }
            (sim_dir / f"optimization_{optimization_id}.json").write_text(
                json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write optimization snapshot: %s", exc)

        note = None
        if fleet_req.required_service_trains > granted:
            note = f"Optimization granted {granted} trains, fewer than required {fleet_req.required_service_trains}."

        return {
            "required_service_trains": fleet_req.required_service_trains,
            "standby_buffer": fleet_req.standby_buffer,
            "total_required_trains": fleet_req.total_required_trains,
            "calculation_method": fleet_req.calculation_method,
            "eligible_train_count": result.eligible_count,
            "granted_train_count": granted,
            "actual_induct_count": granted,
            "service_shortfall": max(0, fleet_req.required_service_trains - granted),
            "note": note,
            "diagnostics": diagnostics,
            "decisions": [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in decisions],
            "stabling_geometry": result.stabling_geometry,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Optimization failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")