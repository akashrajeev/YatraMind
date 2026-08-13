"""Canonical optimization API with compatibility routes."""
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
from app.security import require_role, require_api_key
from app.services.optimization_service import OptimizationService
from app.services.optimization_store import get_latest_decisions, get_decisions_from_history
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.utils.cloud_database import cloud_db_manager
from app.api import optimization_legacy

logger = logging.getLogger(__name__)
router = APIRouter()


def _deterministic_value_from_id(trainset_id: str, field: str | None = None) -> float:
    """Return a stable value, optionally namespaced by a field name."""
    key = f"{field}:{trainset_id}" if field else str(trainset_id)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


@router.post("/run")
async def run_optimization(background_tasks: BackgroundTasks, request: OptimizationRequest, current_user: User = Depends(require_role(UserRole.ADMIN))):
    del current_user
    try:
        trainsets = [dict(item) for item in await MongoTrainsetRepository().list_all()]
        if not trainsets:
            raise HTTPException(status_code=404, detail="No trainsets found")
        result = await OptimizationService().optimize(trainsets, request)
        decisions = result.decisions
        fleet_req = result.fleet_requirement
        granted = result.inducted_count
        fleet_payload = fleet_req.model_dump() if hasattr(fleet_req, "model_dump") else fleet_req.dict()
        decision_payload = [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in decisions]
        diagnostics = {"required_service_trains": fleet_req.required_service_trains, "standby_buffer": fleet_req.standby_buffer, "calculation_method": fleet_req.calculation_method, "eligible_train_count": result.eligible_count, "granted_train_count": granted, "fleet_requirement": fleet_payload}
        await MongoOptimizationRepository().save_run({"timestamp": datetime.utcnow().isoformat(), "target_date": request.target_date.isoformat(), "required_service_count": request.required_service_count, "service_date": request.service_date, "total_decisions": len(decisions), "inducted_count": granted, "standby_count": sum(d.decision == "STANDBY" for d in decisions), "maintenance_count": sum(d.decision == "MAINTENANCE" for d in decisions), "average_confidence": sum(d.confidence_score for d in decisions) / len(decisions) if decisions else 0, "decisions": decision_payload, "fleet_requirement": fleet_payload})
        if hasattr(optimization_legacy, "write_optimization_metrics"):
            background_tasks.add_task(optimization_legacy.write_optimization_metrics, decisions)
        try:
            sim_dir = Path(getattr(settings, "SIMULATION_SAVE_DIR", "backend/simulation_runs"))
            sim_dir.mkdir(parents=True, exist_ok=True)
            optimization_id = str(uuid.uuid4())
            (sim_dir / f"optimization_{optimization_id}.json").write_text(json.dumps({"optimization_id": optimization_id, "timestamp": datetime.utcnow().isoformat(), "diagnostics": diagnostics, "decisions": decision_payload}, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write optimization snapshot: %s", exc)
        note = None if fleet_req.required_service_trains <= granted else f"Optimization granted {granted} trains, fewer than required {fleet_req.required_service_trains}."
        return {"required_service_trains": fleet_req.required_service_trains, "standby_buffer": fleet_req.standby_buffer, "total_required_trains": fleet_req.total_required_trains, "calculation_method": fleet_req.calculation_method, "eligible_train_count": result.eligible_count, "granted_train_count": granted, "actual_induct_count": granted, "service_shortfall": max(0, fleet_req.required_service_trains - granted), "note": note, "diagnostics": diagnostics, "decisions": decision_payload, "stabling_geometry": result.stabling_geometry}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Optimization failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")


async def _load_trainsets() -> list[Dict[str, Any]]:
    return [dict(item) for item in await MongoTrainsetRepository().list_all()]


async def _get_decisions_for_geometry() -> list[Dict[str, Any]] | None:
    decisions = await get_latest_decisions()
    if decisions:
        return decisions
    return await get_decisions_from_history()


@router.get("/stabling-geometry")
async def get_stabling_geometry_optimization(_auth=Depends(require_api_key)):
    try:
        trainsets = await _load_trainsets()
        if not trainsets:
            raise HTTPException(status_code=404, detail="No trainsets found")
        decisions = await _get_decisions_for_geometry()
        if not decisions:
            raise HTTPException(status_code=400, detail={"error": "No optimization decisions available. Run optimization first.", "code": "no_induction_decisions"})
        geometry = await StablingGeometryOptimizer().optimize_stabling_geometry(trainsets, decisions)
        efficiency = geometry.get("efficiency_metrics", {}).get("overall_efficiency")
        geometry["efficiency_improvement"] = round(float(efficiency) * 100, 2) if efficiency is not None else 0.0
        layout = geometry.get("optimized_layout", {})
        geometry["total_optimized_positions"] = sum(len(v.get("bay_assignments", {})) for v in layout.values())
        return geometry
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Stabling geometry failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/shunting-schedule")
async def get_shunting_schedule(_auth=Depends(require_api_key)):
    try:
        trainsets = await _load_trainsets()
        if not trainsets:
            raise HTTPException(status_code=404, detail="No trainsets found")
        decisions = await _get_decisions_for_geometry()
        if not decisions:
            raise HTTPException(status_code=400, detail={"error": "No optimization decisions available. Run optimization first.", "code": "no_induction_decisions"})
        stabling = await StablingGeometryOptimizer().optimize_stabling_geometry(trainsets, decisions)
        operations = stabling.get("shunting_operations", [])
        summary = stabling.get("shunting_summary", {})
        total_time = summary.get("total_time_min", 0)
        available = StablingGeometryOptimizer().operational_window.get("minutes", 120)
        return {"shunting_schedule": operations, "schedule_by_depot": {"Muttom Depot": operations}, "depot_summaries": {"Muttom Depot": summary}, "total_operations": summary.get("total_operations", len(operations)), "estimated_total_time": total_time, "crew_requirements": {"high_complexity": sum(op.get("complexity") == "HIGH" for op in operations), "medium_complexity": sum(op.get("complexity") == "MEDIUM" for op in operations), "low_complexity": sum(op.get("complexity") == "LOW" for op in operations)}, "shunting_window": {"available_minutes": available, "required_minutes": total_time, "buffer_minutes": max(0, available - total_time), "feasible": total_time <= available}, "optimization_timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Shunting schedule failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


for route in optimization_legacy.router.routes:
    if getattr(route, "path", None) not in {"/run", "/stabling-geometry", "/shunting-schedule"}:
        router.routes.append(route)
