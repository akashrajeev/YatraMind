"""
Multi-Depot Simulation API endpoints
"""
import logging
import time
import uuid
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.models.depot import DepotConfig, LocationType
from app.services.simulation.coordinator import run_simulation
from app.services.simulation.infrastructure_advisor import suggest_infrastructure
from app.services.ml_health import check_ai_services_available

router = APIRouter()
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Load depot presets
DEPOTS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "depots.yaml"


def load_depot_presets() -> Dict[str, Any]:
    """Load depot presets from YAML"""
    try:
        with open(DEPOTS_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            return config.get("depots", []), config.get("cost_parameters", {})
    except Exception as e:
        logger.error(f"Failed to load depot config: {e}")
        return [], {}


class DepotRequest(BaseModel):
    """Depot configuration in request"""
    name: str
    location_type: str = "FULL_DEPOT"
    service_bays: int = Field(ge=0)
    maintenance_bays: int = Field(ge=0)
    standby_bays: int = Field(ge=0)
    total_bays: Optional[int] = None
    max_shunting_window_min: int = 120
    is_primary_depot: bool = False
    coordinates: Optional[Dict[str, float]] = None
    enrichment: Optional[Dict[str, Any]] = None


class SimulationRequest(BaseModel):
    """Simulation request payload"""
    depots: List[DepotRequest]
    fleet: int = Field(ge=1, description="Total fleet size")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    service_requirement: Optional[int] = Field(None, description="Required service trains (auto if not provided)")
    ai_mode: bool = Field(True, description="Use AI/ML services if available")
    sim_days: int = Field(1, ge=1, le=7, description="Number of simulation days")


@router.post("/simulate", response_model=Dict[str, Any])
async def simulate_multi_depot(request: SimulationRequest):
    """
    Run multi-depot simulation
    
    Example payload:
    {
      "depots": [
        {"name": "Muttom", "location_type": "FULL_DEPOT", "service_bays": 6, "maintenance_bays": 4, "standby_bays": 2},
        {"name": "Kakkanad", "location_type": "FULL_DEPOT", "service_bays": 6, "maintenance_bays": 3, "standby_bays": 2}
      ],
      "fleet": 40,
      "seed": 12345,
      "service_requirement": 20,
      "ai_mode": true
    }
    """
    run_id = None
    start_time = time.time()
    try:
        # Generate run_id early for error tracking
        run_id = f"SIM_{uuid.uuid4().hex[:12]}"
        
        # Check AI availability
        actually_used_ai = False
        if request.ai_mode:
            ai_available = check_ai_services_available()
            if ai_available:
                actually_used_ai = True
            else:
                logger.warning("AI services unavailable; using deterministic fallback")
        # Convert request depots to DepotConfig
        depot_configs = []
        for depot_req in request.depots:
            try:
                location_type = LocationType(depot_req.location_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid location_type: {depot_req.location_type}. Must be FULL_DEPOT, TERMINAL_YARD, or MAINLINE_SIDING"
                )
            
            depot_config = DepotConfig(
                depot_id=depot_req.name.upper().replace(" ", "_"),
                name=depot_req.name,
                location_type=location_type,
                service_bays=depot_req.service_bays,
                maintenance_bays=depot_req.maintenance_bays,
                standby_bays=depot_req.standby_bays,
                total_bays=depot_req.total_bays,
                max_shunting_window_min=depot_req.max_shunting_window_min,
                is_primary_depot=depot_req.is_primary_depot,
                coordinates=depot_req.coordinates,
                enrichment=depot_req.enrichment
            )
            depot_configs.append(depot_config)
        
        # Run simulation with actual AI status
        result = run_simulation(
            depots=depot_configs,
            fleet_count=request.fleet,
            service_requirement=request.service_requirement,
            seed=request.seed,
            sim_days=request.sim_days,
            ai_mode=actually_used_ai  # Use actual AI status, not requested
        )
        
        # Add fallback warning if AI was requested but not used
        if request.ai_mode and not actually_used_ai:
            result.warnings.append("AI services unavailable; deterministic fallback used")
        
        duration = time.time() - start_time
        
        # Log simulation run
        logger.info(
            f"SIMULATE_RUN run_id={result.run_id} fleet={request.fleet} "
            f"depots={len(depot_configs)} used_ai={actually_used_ai} duration={duration:.2f}"
        )
        
        # Get infrastructure recommendations
        depot_config_map = {d.depot_id: d for d in depot_configs}
        presets, cost_params = load_depot_presets()
        infrastructure_recs = suggest_infrastructure(
            result.per_depot,
            depot_config_map,
            cost_params
        )
        
        # Ensure global_summary has all required keys with defaults
        global_summary = result.global_summary.copy()
        required_keys = {
            "service_trains": 0,
            "required_service": 13,
            "stabled_service": 0,
            "service_shortfall": 0,
            "shunting_time": 0,
            "turnout_time": 0,
            "total_capacity": 0,
            "fleet": request.fleet,
            "transfers_recommended": 0,
            "shunting_feasible": True
        }
        for key, default_value in required_keys.items():
            if key not in global_summary:
                global_summary[key] = default_value
                logger.warning(f"Missing key in global_summary: {key}, using default {default_value}")
        
        # Convert to response format
        response = {
            "run_id": result.run_id,
            "used_ai": actually_used_ai,
            "seed": result.seed,
            "config_snapshot": result.config_snapshot,
            "per_depot": {
                depot_id: {
                    "depot_id": res.depot_id,
                    "depot_name": res.depot_name,
                    "assigned_trains": res.assigned_trains,
                    "stabling_summary": res.stabling_summary,
                    "bay_layout_before": res.bay_layout_before,
                    "bay_layout_after": res.bay_layout_after,
                    "bay_diff": res.bay_diff,
                    "shunting_operations": res.shunting_operations,
                    "shunting_summary": res.shunting_summary,
                    "kpis": res.kpis,
                    "warnings": res.warnings,
                    "violations": res.violations
                }
                for depot_id, res in result.per_depot.items()
            },
            "inter_depot_transfers": [
                {
                    "from_depot": t.from_depot,
                    "to_depot": t.to_depot,
                    "train_id": t.train_id,
                    "cost_estimate": t.cost_estimate,
                    "benefit_estimate": t.benefit_estimate,
                    "reason": t.reason,
                    "feasibility": t.feasibility,
                    "recommended": t.recommended,
                    "dead_km": t.dead_km,
                    "estimated_time_hours": t.estimated_time_hours
                }
                for t in result.inter_depot_transfers
            ],
            "global_summary": global_summary,
            "warnings": result.warnings or [],
            "infrastructure_recommendations": [
                {
                    "depot_id": r.depot_id,
                    "depot_name": r.depot_name,
                    "bay_type": r.bay_type,
                    "bays_to_add": r.bays_to_add,
                    "estimated_cost": r.estimated_cost,
                    "shortfall_reduction": r.shortfall_reduction,
                    "payback_days": r.payback_days,
                    "roi": r.roi
                }
                for r in infrastructure_recs
            ],
            "export_links": {
                "json": f"/api/v1/simulate/{result.run_id}/export/json",
                "pdf": f"/api/v1/simulate/{result.run_id}/export/pdf"
            }
        }
        
        return response
        
    except Exception as e:
        # Log full error
        error_run_id = run_id or f"ERROR_{uuid.uuid4().hex[:8]}"
        logger.exception(f"Simulation error: {e}")
        
        # Write to error log file
        try:
            error_log_path = Path("logs/simulation_errors.log")
            error_log_path.parent.mkdir(exist_ok=True)
            with open(error_log_path, "a") as f:
                from datetime import datetime
                f.write(f"{error_run_id}|{datetime.utcnow().isoformat()}|{str(e)[:200]}\n")
        except Exception as log_error:
            logger.error(f"Failed to write error log: {log_error}")
        
        # Return JSON error response
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Simulation failed",
                "run_id": error_run_id,
                "short_error": str(e)[:200]
            }
        )


@router.get("/simulate/{run_id}", response_model=Dict[str, Any])
async def get_simulation_run(run_id: str):
    """Get historical simulation run details"""
    # TODO: Implement database lookup
    raise HTTPException(status_code=501, detail="Not implemented: database lookup required")


@router.get("/simulate/{run_id}/export/json")
async def export_simulation_json(run_id: str):
    """Export simulation run as JSON"""
    # TODO: Implement JSON export
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/simulate/{run_id}/export/pdf")
async def export_simulation_pdf(run_id: str):
    """Export simulation run as PDF"""
    # TODO: Implement PDF export
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/depots/presets", response_model=Dict[str, Any])
async def get_depot_presets():
    """Get available depot presets"""
    presets, cost_params = load_depot_presets()
    return {
        "depots": presets,
        "cost_parameters": cost_params
    }


@router.get("/stabling", response_model=Dict[str, Any])
async def get_stabling(
    depots: Optional[str] = Query(None, description="Comma-separated depot names"),
    fleet: Optional[int] = Query(None, description="Fleet size"),
    seed: Optional[int] = Query(None, description="Random seed"),
    run_id: Optional[str] = Query(None, description="Existing run ID")
):
    """
    Get stabling geometry (extended to support simulation)
    If run_id provided, return that run. Otherwise, run lightweight simulation.
    """
    if run_id:
        # TODO: Lookup existing run
        raise HTTPException(status_code=501, detail="Run ID lookup not implemented")
    
    if not depots or not fleet:
        raise HTTPException(
            status_code=400,
            detail="depots and fleet parameters required for simulation"
        )
    
    # Parse depots
    depot_names = [d.strip() for d in depots.split(",")]
    presets, _ = load_depot_presets()
    
    # Find matching presets
    depot_configs = []
    for preset in presets:
        if preset.get("name") in depot_names or preset.get("depot_id") in depot_names:
            depot_config = DepotConfig(
                depot_id=preset["depot_id"],
                name=preset["name"],
                location_type=LocationType(preset["location_type"]),
                service_bays=preset["service_bays"],
                maintenance_bays=preset["maintenance_bays"],
                standby_bays=preset["standby_bays"],
                total_bays=preset.get("total_bays"),
                max_shunting_window_min=preset.get("max_shunting_window_min", 120),
                is_primary_depot=preset.get("is_primary_depot", False),
                coordinates=preset.get("coordinates"),
                enrichment=preset.get("enrichment")
            )
            depot_configs.append(depot_config)
    
    if not depot_configs:
        raise HTTPException(status_code=404, detail="No matching depot presets found")
    
    # Run lightweight simulation
    result = run_simulation(
        depots=depot_configs,
        fleet_count=fleet,
        seed=seed,
        ai_mode=False  # Lightweight mode
    )
    
    return {
        "run_id": result.run_id,
        "per_depot": {
            depot_id: {
                "stabling_summary": res.stabling_summary,
                "bay_layout_after": res.bay_layout_after
            }
            for depot_id, res in result.per_depot.items()
        },
        "global_summary": result.global_summary
    }

