"""Canonical stabling and depot domain types."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class LocationType(str, Enum):
    FULL_DEPOT = "FULL_DEPOT"
    TERMINAL_YARD = "TERMINAL_YARD"
    MAINLINE_SIDING = "MAINLINE_SIDING"


class BayRole(str, Enum):
    SERVICE = "SERVICE"
    STANDBY = "STANDBY"
    MAINTENANCE = "MAINTENANCE"


@dataclass(frozen=True)
class Bay:
    bay_id: str
    role: BayRole
    capacity: int = 1
    distance_to_exit_m: float = 0.0
    turnout_time_min: int = 0


@dataclass(frozen=True)
class Depot:
    depot_name: str
    location_type: LocationType
    service_bay_capacity: int
    maintenance_bay_capacity: int
    supports_heavy_maintenance: bool = True
    supports_cleaning: bool = True
    can_start_service: bool = True


@dataclass(frozen=True)
class StablingAssignment:
    trainset_id: str
    depot_name: str
    bay_id: str
    role: BayRole
    dead_km_in: float = 0.0
    dead_km_out: float = 0.0
    placement_reason_code: str = "DEFAULT"
