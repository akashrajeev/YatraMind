from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class TrainsetStatus(str, Enum):
    ACTIVE = "ACTIVE"
    STANDBY = "STANDBY"
    MAINTENANCE = "MAINTENANCE"


class LocationType(str, Enum):
    FULL_DEPOT = "FULL_DEPOT"
    TERMINAL_YARD = "TERMINAL_YARD"
    MAINLINE_SIDING = "MAINLINE_SIDING"


class MaintenanceSeverity(str, Enum):
    NONE = "NONE"
    LIGHT = "LIGHT"
    HEAVY = "HEAVY"


class JobCards(BaseModel):
    open_cards: int = 0
    critical_cards: int = 0


class Trainset(BaseModel):
    trainset_id: str
    status: TrainsetStatus = TrainsetStatus.STANDBY
    current_location: Dict[str, Any] = Field(default_factory=lambda: {"depot": "KALAMASSERY", "bay": ""})
    current_mileage: float = 0.0
    max_mileage_before_maintenance: float = 50000.0
    fitness_certificates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    job_cards: JobCards = Field(default_factory=JobCards)
    branding_priority: int = 0
    sensor_health_score: float = 1.0
    maintenance_severity: MaintenanceSeverity = Field(default=MaintenanceSeverity.NONE, description="Maintenance severity flag")


class TrainsetUpdate(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict)


class OptimizationWeights(BaseModel):
    """Customizable weights for optimization factors"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "readiness": 0.35,
                "reliability": 0.30,
                "branding": 0.20,
                "shunt": 0.10,
                "mileage_balance": 0.05
            }
        }
    )
    
    readiness: float = Field(0.35, ge=0.0, le=1.0, description="Weight for service readiness (0-1)")
    reliability: float = Field(0.30, ge=0.0, le=1.0, description="Weight for reliability/health (0-1)")
    branding: float = Field(0.20, ge=0.0, le=1.0, description="Weight for branding priority (0-1)")
    shunt: float = Field(0.10, ge=0.0, le=1.0, description="Weight for shunt cost minimization (0-1)")
    mileage_balance: float = Field(0.05, ge=0.0, le=1.0, description="Weight for mileage balance (0-1)")


class OptimizationRequest(BaseModel):
    target_date: datetime = Field(default_factory=datetime.utcnow)
    service_date: Optional[str] = Field(
        default=None,
        description="Date for timetable lookup (YYYY-MM-DD). If not provided, uses default timetable configuration.",
    )
    required_service_count: Optional[int] = Field(
        default=None,
        description="Manual override: explicit train count request. If provided, overrides timetable calculation.",
    )
    override_constraints: Optional[Dict[str, Any]] = None
    weights: Optional[OptimizationWeights] = Field(
        default=None,
        description="Custom weights for optimization factors. If not provided, defaults will be used."
    )


class ShapFeature(BaseModel):
    name: str
    value: float
    impact: str  # "positive" | "negative" | "neutral"


class InductionDecision(BaseModel):
    trainset_id: str
    decision: str  # INDUCT | STANDBY | MAINTENANCE
    confidence_score: float = 0.8
    reasons: List[str] = Field(default_factory=list)
    # Enhanced explainability fields
    score: float = Field(default=0.0, description="Composite score for this assignment")
    top_reasons: List[str] = Field(default_factory=list, description="Top 3 contributing positive reasons")
    top_risks: List[str] = Field(default_factory=list, description="Top 3 negative reasons")
    violations: List[str] = Field(default_factory=list, description="List of rule violations if assignment chosen despite violation")
    shap_values: List[ShapFeature] = Field(default_factory=list, description="Top 5 features and their impact if ML used")


# Stabling Geometry Models
class FleetSummary(BaseModel):
    """Fleet-level summary statistics"""
    total_trainsets: int
    required_service_trains: int
    standby_buffer: int
    total_required_trains: int
    eligible_count: int
    actual_induct_count: int
    actual_standby_count: int
    maintenance_count: int
    service_shortfall: int = Field(default=0, description="Number of trains short if requirement not met")
    compliance_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Compliance rate (0-1)")
    standby_at_muttom: int = Field(default=0, description="Standby rakes currently at Muttom depot")


class DepotAllocation(BaseModel):
    """Depot-level allocation breakdown"""
    depot_name: str
    service_trains: int
    standby_trains: int
    maintenance_trains: int
    total_trains: int
    service_bay_capacity: int
    maintenance_bay_capacity: int
    total_bay_capacity: int
    capacity_warning: bool = Field(default=False, description="True if any category exceeds capacity")
    location_type: Optional[str] = Field(default=None, description="FULL_DEPOT | TERMINAL_YARD | MAINLINE_SIDING")
    supports_heavy_maintenance: Optional[bool] = None
    supports_cleaning: Optional[bool] = None
    can_start_service: Optional[bool] = None


class BayAssignment(BaseModel):
    """Individual bay assignment with details"""
    bay_id: int
    role: str = Field(..., description="SERVICE | STANDBY | MAINTENANCE")
    trainset_id: Optional[str] = Field(None, description="Trainset assigned to this bay, None if empty")
    turnout_time_min: Optional[int] = Field(None, description="Time to exit bay in minutes")
    distance_to_exit_m: Optional[int] = Field(None, description="Distance to depot exit in meters")
    notes: Optional[str] = Field(None, description="Additional context (branding, job cards, etc.)")
    reason_code: Optional[str] = Field(None, description="Placement rationale e.g. DEADKM_OPTIMIZED, MAINT_DEPOT")
    dead_km_in: Optional[float] = Field(None, description="Dead kilometres inbound to stabling location")
    dead_km_out: Optional[float] = Field(None, description="Dead kilometres outbound to first departure")
    first_departure_station: Optional[str] = Field(None, description="First departure station for this rake")
    stabled_at: Optional[str] = Field(None, description="Location where the train is stabled")
    placement_reason_code: Optional[str] = Field(None, description="MAINT_DEPOT | DEADKM_MIN | MUTTOM_STANDBY_BUFFER | WRAP_SLA | DEFAULT")
    placement_reason_text: Optional[str] = Field(None, description="Human-readable summary of placement rationale")
    dead_km: Optional[Dict[str, float]] = Field(default=None, description="Dead km breakdown {in,out,total}")


class OptimizationKPIs(BaseModel):
    """Optimization performance metrics"""
    optimized_positions: int
    total_shunting_time_min: int
    total_turnout_time_min: int
    efficiency_improvement_pct: float = Field(default=0.0, description="Efficiency improvement percentage")
    energy_savings_kwh: Optional[float] = Field(None, description="Estimated energy savings in kWh")
    night_movements_reduced: Optional[int] = Field(None, description="Number of movements reduced vs baseline")


class StablingGeometryResponse(BaseModel):
    """Rich stabling geometry response with structured intelligence"""
    fleet_summary: FleetSummary
    depot_allocation: List[DepotAllocation]
    bay_layout: Dict[str, List[BayAssignment]] = Field(..., description="Depot name -> list of bay assignments")
    optimization_kpis: OptimizationKPIs
    warnings: List[str] = Field(default_factory=list, description="Capacity or operational warnings")
    optimization_timestamp: str = Field(..., description="ISO timestamp of optimization")
    depot_usage: Optional[Dict[str, Any]] = Field(default=None, description="Aggregated depot usage breakdown")
    shunting_operations: Optional[List[Dict[str, Any]]] = None
    capacity_summary: Optional[Dict[str, Any]] = None
    unassigned_trainsets: Optional[List[Dict[str, Any]]] = None
    maintenance_queue: Optional[List[Dict[str, Any]]] = None
    shunting_window: Optional[Dict[str, Any]] = None
    service_requirement: Optional[Dict[str, Any]] = None
    induction_summary: Optional[Dict[str, Any]] = None
    stabling_summary: Optional[Dict[str, Any]] = None
