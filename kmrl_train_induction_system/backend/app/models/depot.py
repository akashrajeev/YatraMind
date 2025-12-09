"""
Depot configuration and data models for multi-depot simulation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LocationType(str, Enum):
    FULL_DEPOT = "FULL_DEPOT"
    TERMINAL_YARD = "TERMINAL_YARD"
    MAINLINE_SIDING = "MAINLINE_SIDING"


class DepotConfig(BaseModel):
    """Depot configuration model"""
    depot_id: str
    name: str
    location_type: LocationType
    service_bays: int = Field(ge=0)
    maintenance_bays: int = Field(ge=0)
    standby_bays: int = Field(ge=0)
    total_bays: Optional[int] = None
    max_shunting_window_min: int = Field(default=120, ge=0)
    is_primary_depot: bool = False
    coordinates: Optional[Dict[str, float]] = None
    display_order: int = 0
    enrichment: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        """Compute total_bays if not provided"""
        if self.total_bays is None:
            self.total_bays = self.compute_total_bays()

    def compute_total_bays(self) -> int:
        """Compute total bays from components"""
        return self.service_bays + self.maintenance_bays + self.standby_bays

    def get_turnout_time(self, bay_id: str) -> int:
        """Get turnout time for a specific bay"""
        if self.enrichment and "turnout_time_map" in self.enrichment:
            return self.enrichment["turnout_time_map"].get(bay_id, 5)  # Default 5 min
        return 5

    def get_capacity_summary(self) -> Dict[str, int]:
        """Get capacity summary"""
        return {
            "service_bays": self.service_bays,
            "maintenance_bays": self.maintenance_bays,
            "standby_bays": self.standby_bays,
            "total_bays": self.total_bays or self.compute_total_bays()
        }


@dataclass
class DepotSimulationResult:
    """Result of simulating a single depot"""
    depot_id: str
    depot_name: str
    assigned_trains: List[str]  # List of train IDs
    stabling_summary: Dict[str, Any]
    bay_layout_before: Dict[str, Any]
    bay_layout_after: Dict[str, Any]
    bay_diff: List[Dict[str, Any]]
    shunting_operations: List[Dict[str, Any]]
    shunting_summary: Dict[str, Any]
    kpis: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)


@dataclass
class TransferRecommendation:
    """Inter-depot transfer recommendation"""
    from_depot: str
    to_depot: str
    train_id: str
    cost_estimate: float
    benefit_estimate: float
    reason: str
    feasibility: bool = True
    recommended: bool = False
    dead_km: float = 0.0
    estimated_time_hours: float = 0.0


@dataclass
class SimulationResult:
    """Complete simulation result"""
    run_id: str
    seed: Optional[int]
    config_snapshot: Dict[str, Any]
    per_depot: Dict[str, DepotSimulationResult]
    inter_depot_transfers: List[TransferRecommendation]
    global_summary: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    export_links: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[str] = None

