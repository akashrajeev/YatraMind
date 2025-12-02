"""What-If Simulation API endpoints"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

from app.services.whatif_simulator import run_whatif, SIMULATION_RUNS_DIR
from app.utils.snapshot import capture_snapshot
from app.security import require_api_key
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


class WhatIfScenario(BaseModel):
    """What-If scenario configuration"""
    required_service_hours: Optional[int] = Field(None, description="Override service hours requirement")
    override_train_attributes: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, 
        description="Path-based nested setter (e.g., {'T-001': {'fitness.telecom.valid_until': '2024-12-31'}})"
    )
    depot_layout_override: Optional[Dict[str, Any]] = Field(None, description="Override depot layouts")
    cleaning_capacity_override: Optional[Dict[str, Any]] = Field(None, description="Override cleaning capacity")
    force_decisions: Optional[Dict[str, str]] = Field(
        None,
        description="Force specific trainset decisions (e.g., {'T-001': 'INDUCT', 'T-002': 'MAINTENANCE'})"
    )
    inject_delay_events: Optional[list] = Field(None, description="Inject delay events")
    random_seed: Optional[int] = Field(None, description="Random seed for deterministic execution")
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optimization weights (e.g., {'readiness': 0.35, 'reliability': 0.30, 'branding': 0.20, 'shunt': 0.10, 'mileage_balance': 0.05})"
    )


def _normalize_decision_explain(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure decision has an 'explain' field (defensive backend check)"""
    if not isinstance(decision, dict):
        return decision
    
    if "explain" in decision and decision["explain"]:
        return decision
    
    # Synthesize explanation from available fields
    explanation_parts = []
    decision_type = decision.get("decision", decision.get("role", "UNKNOWN"))
    explanation_parts.append(f"Decision: {decision_type}")
    
    if decision.get("reasons"):
        reasons = decision["reasons"]
        if isinstance(reasons, list):
            explanation_parts.extend([str(r) for r in reasons])
        else:
            explanation_parts.append(str(reasons))
    
    confidence = decision.get("confidence_score", 0)
    if confidence:
        explanation_parts.append(f"Confidence: {confidence:.0%}")
    
    decision["explain"] = ". ".join(explanation_parts) if explanation_parts else "No explanation available"
    return decision


def _ensure_results_is_array(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure results field is always an array and normalize decision explain fields"""
    if "results" in data:
        results = data["results"]
        if not isinstance(results, list):
            if results:
                # Convert object to array
                data["results"] = [results]
            else:
                # Empty if falsy
                data["results"] = []
        
        # Normalize each result's decisions to have explain fields
        for result in data["results"]:
            if isinstance(result, dict):
                # Normalize decisions array if present
                if "decisions" in result and isinstance(result["decisions"], list):
                    result["decisions"] = [_normalize_decision_explain(d) for d in result["decisions"]]
                # Also normalize if result itself is a decision
                if "decision" in result or "role" in result:
                    _normalize_decision_explain(result)
    else:
        data["results"] = []
    return data


@router.post("/run")
async def run_simulation(
    scenario: WhatIfScenario,
    _auth=Depends(require_api_key),
):
    """Run What-If simulation immediately and return results"""
    try:
        logger.info("Running What-If simulation via API")
        
        # Convert Pydantic model to dict
        scenario_dict = scenario.dict(exclude_none=True)
        
        # Run simulation
        result = await run_whatif(scenario_dict)
        
        # Ensure results is always an array
        result = _ensure_results_is_array(result)
        
        logger.info(f"Simulation {result['simulation_id']} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Simulation API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.get("/result/{simulation_id}")
async def get_simulation_result(
    simulation_id: str,
    _auth=Depends(require_api_key),
):
    """Load saved simulation result by ID"""
    try:
        result_file = SIMULATION_RUNS_DIR / f"{simulation_id}.json"
        
        if not result_file.exists():
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Ensure results is always an array
        data = _ensure_results_is_array(data)
        
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading simulation result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading simulation: {str(e)}")


@router.get("/snapshot")
async def get_snapshot(
    _auth=Depends(require_api_key),
):
    """Get current system snapshot"""
    try:
        snapshot = await capture_snapshot()
        return snapshot
        
    except Exception as e:
        logger.error(f"Error capturing snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error capturing snapshot: {str(e)}")

