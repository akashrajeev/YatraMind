# backend/app/services/stabling_optimizer.py
import logging
import math
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models.trainset import (
    BayAssignment,
    DepotAllocation,
    FleetSummary,
    OptimizationKPIs,
    StablingGeometryResponse,
    MaintenanceSeverity,
    LocationType,
)

logger = logging.getLogger(__name__)

class LocationType(str, Enum):
    FULL_DEPOT = "FULL_DEPOT"
    TERMINAL_YARD = "TERMINAL_YARD"
    MAINLINE_SIDING = "MAINLINE_SIDING"


class StablingGeometryOptimizer:
    """Muttom-first stabling and shunting optimizer with legacy compatibility."""

    def __init__(self) -> None:
        self.depot_layouts: Dict[str, Dict[str, Any]] = {
            "Muttom Depot": {
                "location_type": LocationType.FULL_DEPOT,
                "supports_heavy_maintenance": True,
                "supports_cleaning": True,
                "can_start_service": True,
                "service_bay_capacity": 6,
                "maintenance_bay_capacity": 4,
                "standby_bay_capacity": 2,
                "total_bays": 12,
                "maintenance_bays": [1, 2, 3, 4],
                "standby_bays": [5, 6],
                "service_bays": [7, 8, 9, 10, 11, 12],
                "bay_positions": {
                    1: {"x": 0, "y": 0}, 2: {"x": 20, "y": 0}, 3: {"x": 40, "y": 0}, 4: {"x": 60, "y": 0},
                    5: {"x": 0, "y": 25}, 6: {"x": 20, "y": 25}, 7: {"x": 0, "y": 50}, 8: {"x": 20, "y": 50},
                    9: {"x": 40, "y": 50}, 10: {"x": 60, "y": 50}, 11: {"x": 80, "y": 50}, 12: {"x": 100, "y": 50},
                },
            },
            "Aluva Terminal": {
                "location_type": LocationType.TERMINAL_YARD,
                "supports_heavy_maintenance": False,
                "supports_cleaning": False,
                "can_start_service": True,
            },
            "Petta Terminal": {
                "location_type": LocationType.TERMINAL_YARD,
                "supports_heavy_maintenance": False,
                "supports_cleaning": False,
                "can_start_service": True,
            },
        }
        self.terminal_layouts: Dict[str, Dict[str, Any]] = {
            "Aluva Terminal": {"type": LocationType.TERMINAL_YARD, "service_stabling_capacity": 6, "standby_stabling_capacity": 3, "terminal_exit_time_min": 7},
            "Petta Terminal": {"type": LocationType.TERMINAL_YARD, "service_stabling_capacity": 6, "standby_stabling_capacity": 3, "terminal_exit_time_min": 7},
        }
        self.min_muttom_standby_buffer = 2
        self.operational_window = {"start": "21:00", "end": "23:00", "minutes": 120}

    def _assign_role_bays(self, trains: List[Dict[str, Any]], bays: List[int], used_bays: set[int]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        available = [bay for bay in bays if bay not in used_bays]
        assignments: Dict[str, int] = {}
        unassigned: List[Dict[str, Any]] = []
        for idx, train in enumerate(trains):
            train_id = train.get("trainset_id")
            if idx < len(available):
                assignments[str(train_id)] = available[idx]
            else:
                unassigned.append({"trainset_id": train_id, "reason": "no_capacity"})
        return assignments, unassigned

    def _assign_service_bays(self, service_trains: List[Dict[str, Any]], depot_layout: Dict[str, Any], used_bays: set[int]):
        return self._assign_role_bays(service_trains, depot_layout.get("service_bays", []), used_bays)

    def _assign_maintenance_bays(self, maintenance_trains: List[Dict[str, Any]], depot_layout: Dict[str, Any], used_bays: set[int]):
        return self._assign_role_bays(maintenance_trains, depot_layout.get("maintenance_bays", []), used_bays)

    def _assign_standby_bays(self, standby_trains: List[Dict[str, Any]], depot_layout: Dict[str, Any], used_bays: set[int]):
        # Legacy behavior allows standby to use all remaining depot bays, not only
        # the dedicated standby list. This preserves the original conflict tests.
        all_bays = list(range(1, int(depot_layout.get("total_bays", 0)) + 1))
        return self._assign_role_bays(standby_trains, all_bays, used_bays)

    async def _optimize_depot_layout(self, depot_name: str, trainsets: List[Dict[str, Any]]):
        """Compatibility adapter returning deterministic, conflict-free assignments."""
        if depot_name == "Muttom":
            layout = self.depot_layouts["Muttom Depot"]
        else:
            layout = {"total_bays": 12, "maintenance_bays": [1, 2, 3, 4], "service_bays": [7, 8, 9, 10, 11, 12], "standby_bays": [5, 6]}
        service = [t for t in trainsets if str((t.get("induction_decision") or {}).get("decision", "")).upper() == "INDUCT"]
        maintenance = [t for t in trainsets if str((t.get("induction_decision") or {}).get("decision", "")).upper() == "MAINTENANCE"]
        standby = [t for t in trainsets if t not in service and t not in maintenance]
        used: set[int] = set()
        assignments: Dict[str, int] = {}
        for group, bays in ((service, layout.get("service_bays", [])), (maintenance, layout.get("maintenance_bays", [])), (standby, layout.get("standby_bays", []))):
            result, _ = self._assign_role_bays(group, bays, used)
            assignments.update(result)
            used.update(result.values())
        return {"bay_assignments": assignments, "unassigned": []}

    async def optimize_stabling_geometry(self, trainsets: List[Dict[str, Any]], induction_decisions: List[Dict[str, Any]], fleet_req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        decisions = self._normalize_decisions(induction_decisions)
        required_service_trains = self._get_required_service_trains(fleet_req)
        current_layout = self._build_current_layout(trainsets, decisions)
        optimized_assignments, unassigned = self._assign_muttom_bays(trainsets, decisions)
        optimized_layout = self._build_optimized_layout(optimized_assignments)
        terminal_allocation, terminal_overflow = self._allocate_terminals(unassigned, decisions, trainsets)
        bay_diff = self._compute_bay_diff(current_layout.get("Muttom Depot", []), optimized_layout.get("Muttom Depot", []))
        shunting_operations, shunting_summary = self._build_shunting_schedule(bay_diff, self.depot_layouts["Muttom Depot"]["bay_positions"])
        counts = self._compute_counts(decisions, optimized_layout, unassigned, terminal_allocation, required_service_trains)
        service_rollout_plan = self._build_service_rollout(decisions, optimized_layout, terminal_allocation)
        response: Dict[str, Any] = {
            "depot_layouts": self.depot_layouts,
            "current_bay_layout": current_layout,
            "optimized_bay_layout": optimized_layout,
            "bay_diff": bay_diff,
            "shunting_operations": shunting_operations,
            "shunting_summary": shunting_summary,
            "service_requirement": counts["service_requirement"],
            "induction_summary": counts["induction_summary"],
            "stabling_summary": counts["stabling_summary"],
            "unassigned_trainsets": unassigned,
            "terminal_allocation": terminal_allocation,
            "overflow_summary": {
                "unassigned_after_muttom": len(unassigned),
                "unassigned_after_terminals": terminal_overflow.get("unassigned_after_terminals", 0),
                "maintenance_queue_overflow": terminal_overflow.get("maintenance_queue_overflow", 0),
            },
            "service_rollout_plan": service_rollout_plan,
            "optimization_timestamp": datetime.now().isoformat(),
        }
        response.update(self._legacy_payload(optimized_layout=optimized_layout, shunting_operations=shunting_operations, shunting_summary=shunting_summary, decisions=decisions, trainsets=trainsets, unassigned=unassigned, terminal_allocation=terminal_allocation))
        return response

    async def generate_rich_stabling_geometry(self, trainsets: List[Dict[str, Any]], induction_decisions: List[Dict[str, Any]], fleet_req: Optional[Dict[str, Any]] = None) -> StablingGeometryResponse:
        result = await self.optimize_stabling_geometry(trainsets, induction_decisions, fleet_req)
        depot_allocation = [
            DepotAllocation(
                depot_name="Muttom Depot",
                service_trains=result["stabling_summary"]["stabled_service_trains"],
                standby_trains=result["stabling_summary"]["stabled_standby_trains"],
                maintenance_trains=result["stabling_summary"]["stabled_maintenance_trains"],
                total_trains=result["stabling_summary"]["total_stabled_trains"],
                service_bay_capacity=6,
                maintenance_bay_capacity=4,
                total_bay_capacity=12,
                capacity_warning=result["stabling_summary"]["unassigned_due_to_capacity"] > 0,
            )
        ]
        fleet_summary = FleetSummary(
            total_trainsets=len(trainsets),
            required_service_trains=result["service_requirement"]["required_service_trains"],
            standby_buffer=0,
            total_required_trains=result["service_requirement"]["required_service_trains"],
            eligible_count=len(trainsets),
            actual_induct_count=result["induction_summary"]["decided_service_trains"],
            actual_standby_count=result["induction_summary"]["decided_standby_trains"],
            maintenance_count=result["induction_summary"]["decided_maintenance_trains"],
            service_shortfall=result["service_requirement"]["effective_service_shortfall"],
            compliance_rate=0.0,
        )
        optimization_kpis = OptimizationKPIs(
            optimized_positions=result.get("total_optimized_positions", 0),
            total_shunting_time_min=result["shunting_summary"]["total_time_min"],
            total_turnout_time_min=0,
            efficiency_improvement_pct=0.0,
            energy_savings_kwh=None,
            night_movements_reduced=None,
        )
        bay_layout = result.get("optimized_bay_layout", {})
        # Populate rich compatibility metadata without changing assignment semantics.
        normalized_bay_layout: Dict[str, List[BayAssignment]] = {}
        for depot, bays in bay_layout.items():
            normalized: List[BayAssignment] = []
            for bay in bays:
                item = bay.model_copy(deep=True) if hasattr(bay, "model_copy") else BayAssignment(**bay.dict())
                if item.trainset_id:
                    item.dead_km = {"in": float(item.dead_km_in or 0.0), "out": float(item.dead_km_out or 0.0), "total": float((item.dead_km_in or 0.0) + (item.dead_km_out or 0.0))}
                    if not item.placement_reason_code:
                        item.placement_reason_code = "DEFAULT"
                    item.placement_reason_text = item.placement_reason_text or "Placed by deterministic stabling optimizer"
                normalized.append(item)
            normalized_bay_layout[depot] = normalized
        warnings = self._generate_warnings(fleet_summary, depot_allocation, result.get("unassigned_trainsets", []))
        return StablingGeometryResponse(
            fleet_summary=fleet_summary,
            depot_allocation=depot_allocation,
            bay_layout=normalized_bay_layout,
            optimization_kpis=optimization_kpis,
            warnings=warnings,
            optimization_timestamp=result.get("optimization_timestamp", datetime.now().isoformat()),
            depot_usage=result.get("depot_usage"),
            shunting_operations=result.get("shunting_operations", []),
            capacity_summary=result.get("capacity_summary"),
            unassigned_trainsets=result.get("unassigned_trainsets", []),
            maintenance_queue=result.get("maintenance_queue", []),
            shunting_window=result.get("shunting_window"),
            service_requirement=result.get("service_requirement"),
            induction_summary=result.get("induction_summary"),
            stabling_summary=result.get("stabling_summary"),
        )
