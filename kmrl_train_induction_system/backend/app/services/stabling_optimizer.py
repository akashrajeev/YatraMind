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
    """
    Final Muttom-first stabling + shunting optimizer.
    - Enforces physical bay capacities (6 service, 4 maintenance, 2 standby).
    - Generates before/after/diff layouts.
    - Shunting schedule derives directly from the diff with feasibility checks.
    """

    def __init__(self) -> None:
        # Single modeled depot (Muttom) plus terminal metadata for completeness
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
                # Simple geometry grid (meters) for shunting distance estimation
                "bay_positions": {
                    1: {"x": 0, "y": 0},
                    2: {"x": 20, "y": 0},
                    3: {"x": 40, "y": 0},
                    4: {"x": 60, "y": 0},
                    5: {"x": 0, "y": 25},
                    6: {"x": 20, "y": 25},
                    7: {"x": 0, "y": 50},
                    8: {"x": 20, "y": 50},
                    9: {"x": 40, "y": 50},
                    10: {"x": 60, "y": 50},
                    11: {"x": 80, "y": 50},
                    12: {"x": 100, "y": 50},
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

        # Terminal yard capacities (siding slots, not bays)
        self.terminal_layouts: Dict[str, Dict[str, Any]] = {
            "Aluva Terminal": {
                "type": LocationType.TERMINAL_YARD,
                "service_stabling_capacity": 6,
                "standby_stabling_capacity": 3,
                "terminal_exit_time_min": 7,
            },
            "Petta Terminal": {
                "type": LocationType.TERMINAL_YARD,
                "service_stabling_capacity": 6,
                "standby_stabling_capacity": 3,
                "terminal_exit_time_min": 7,
            },
        }

        # Minimum standby preferred near Muttom yard (without bay)
        self.min_muttom_standby_buffer = 2

        self.operational_window = {"start": "21:00", "end": "23:00", "minutes": 120}

    async def optimize_stabling_geometry(
        self,
        trainsets: List[Dict[str, Any]],
                                induction_decisions: List[Dict[str, Any]],
        fleet_req: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint that enforces Muttom capacity, produces bay diffs,
        and derives a shunting schedule consistent with those diffs.
        """
        decisions = self._normalize_decisions(induction_decisions)
        required_service_trains = self._get_required_service_trains(fleet_req)

        current_layout = self._build_current_layout(trainsets, decisions)
        optimized_assignments, unassigned = self._assign_muttom_bays(trainsets, decisions)
        optimized_layout = self._build_optimized_layout(optimized_assignments)

        terminal_allocation, terminal_overflow = self._allocate_terminals(unassigned, decisions, trainsets)

        bay_diff = self._compute_bay_diff(
            current_layout.get("Muttom Depot", []),
            optimized_layout.get("Muttom Depot", []),
        )
        shunting_operations, shunting_summary = self._build_shunting_schedule(
            bay_diff, self.depot_layouts["Muttom Depot"]["bay_positions"]
        )

        counts = self._compute_counts(
            decisions,
            optimized_layout,
            unassigned,
            terminal_allocation,
            required_service_trains,
        )

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

        # Backward-compatible fields expected by legacy UI/tests
        response.update(
            self._legacy_payload(
                optimized_layout=optimized_layout,
                shunting_operations=shunting_operations,
                shunting_summary=shunting_summary,
                decisions=decisions,
                trainsets=trainsets,
                unassigned=unassigned,
                terminal_allocation=terminal_allocation,
            )
        )
        return response

    async def generate_rich_stabling_geometry(
        self,
        trainsets: List[Dict[str, Any]],
        induction_decisions: List[Dict[str, Any]],
        fleet_req: Optional[Dict[str, Any]] = None,
    ) -> StablingGeometryResponse:
        """
        Retain the rich response model while reusing the new single-depot optimizer.
        """
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
        warnings = self._generate_warnings(fleet_summary, depot_allocation, result.get("unassigned_trainsets", []))

        return StablingGeometryResponse(
            fleet_summary=fleet_summary,
            depot_allocation=depot_allocation,
            bay_layout=bay_layout,
            optimization_kpis=optimization_kpis,
            warnings=warnings,
            optimization_timestamp=result.get("optimization_timestamp", datetime.now().isoformat()),
        )

    def _normalize_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            decision_raw = d.get("decision") or d.get("role") or "STANDBY"
            decision = (
                "SERVICE"
                if decision_raw in ("INDUCT", "SERVICE")
                else "MAINTENANCE"
                if decision_raw == "MAINTENANCE"
                else "STANDBY"
            )
            normalized.append(
                {
                    "trainset_id": d.get("trainset_id"),
                    "decision": decision,
                    "maintenance_severity": d.get("maintenance_severity", "LIGHT"),
                    "score": d.get("score", 0),
                    "priority": d.get("priority", 0),
                    "first_departure_station": d.get("first_departure_station"),
                    "last_arrival_station": d.get("last_arrival_station"),
                }
            )
        return normalized

    def _get_required_service_trains(self, fleet_req: Optional[Dict[str, Any]]) -> int:
        if fleet_req and isinstance(fleet_req, dict):
            return int(fleet_req.get("required_service_trains") or fleet_req.get("required_service_count") or 13)
        return 13

    def _build_current_layout(
        self, trainsets: List[Dict[str, Any]], decisions: List[Dict[str, Any]]
    ) -> Dict[str, List[BayAssignment]]:
        decision_map = {d["trainset_id"]: d for d in decisions if d.get("trainset_id")}
        bays = [self._empty_bay(bay_id) for bay_id in range(1, 13)]
        bay_map: Dict[int, BayAssignment] = {b.bay_id: b for b in bays}

        for t in trainsets:
            if (t.get("current_location") or {}).get("depot") not in ["Muttom", "Muttom Depot"]:
                continue
            bay_id = self._extract_bay_number((t.get("current_location") or {}).get("bay", ""))
            if not bay_id or bay_id not in bay_map:
                continue
            decision = decision_map.get(t.get("trainset_id"), {})
            role = self._role_from_decision(decision.get("decision"))
            bay_map[bay_id] = BayAssignment(
                bay_id=bay_id,
                role=role,
                trainset_id=t.get("trainset_id"),
                turnout_time_min=None,
                distance_to_exit_m=None,
                notes=None,
            )

        return {"Muttom Depot": list(bay_map.values())}

    def _assign_muttom_bays(
        self, trainsets: List[Dict[str, Any]], decisions: List[Dict[str, Any]]
    ) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
        decision_map = {d["trainset_id"]: d for d in decisions if d.get("trainset_id")}
        maintenance: List[Dict[str, Any]] = []
        service: List[Dict[str, Any]] = []
        standby: List[Dict[str, Any]] = []

        for t in trainsets:
            train_id = t.get("trainset_id")
            dec = decision_map.get(train_id, {"decision": "STANDBY"})
            entry = {**t, "decision": dec.get("decision"), "maintenance_severity": dec.get("maintenance_severity", "LIGHT")}
            entry["score"] = dec.get("score", 0)
            entry["priority"] = dec.get("priority", 0)
            entry["current_mileage"] = t.get("current_mileage", 0)
            entry["job_cards"] = t.get("job_cards", {})
            if dec.get("decision") == "MAINTENANCE":
                maintenance.append(entry)
            elif dec.get("decision") == "SERVICE":
                service.append(entry)
            else:
                standby.append(entry)

        maintenance.sort(
            key=lambda t: (
                0 if str(t.get("maintenance_severity", "")).upper() == "HEAVY" else 1,
                -(t.get("job_cards", {}).get("critical_cards", 0)),
                -(t.get("current_mileage", 0)),
            )
        )
        service.sort(key=lambda t: -t.get("score", 0))
        standby.sort(key=lambda t: -(t.get("priority", 0)))

        layout = self.depot_layouts["Muttom Depot"]
        assignments: Dict[int, Dict[str, Any]] = {}
        unassigned: List[Dict[str, Any]] = []

        def _fill(bays: List[int], trains: List[Dict[str, Any]], role: str) -> None:
            for idx, train in enumerate(trains):
                if idx < len(bays):
                    assignments[bays[idx]] = {"trainset_id": train.get("trainset_id"), "role": role}
                else:
                    unassigned.append(
                        {
                            "trainset_id": train.get("trainset_id"),
                            "decision": role,
                            "reason": "NO_BAY_CAPACITY",
                        }
                    )

        _fill(layout["maintenance_bays"], maintenance, "MAINTENANCE")
        _fill(layout["standby_bays"], standby, "STANDBY")
        _fill(layout["service_bays"], service, "SERVICE")

        # Global cap safety
        if len(assignments) > layout["total_bays"]:
            overflow = list(assignments.items())[layout["total_bays"] :]
            for bay_id, info in overflow:
                unassigned.append({"trainset_id": info["trainset_id"], "decision": info["role"], "reason": "NO_BAY_CAPACITY"})
                assignments.pop(bay_id, None)
        
        return assignments, unassigned
    
    def _allocate_terminals(
        self,
        unassigned_after_muttom: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        trainsets: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        decision_map = {d["trainset_id"]: d for d in decisions if d.get("trainset_id")}
        train_map = {t.get("trainset_id"): t for t in trainsets}

        terminals = {
            name: {
                "service_trains": [],
                "standby_trains": [],
                "service_capacity": layout["service_stabling_capacity"],
                "standby_capacity": layout["standby_stabling_capacity"],
                "service_used": 0,
                "standby_used": 0,
            }
            for name, layout in self.terminal_layouts.items()
        }

        service_overflow: List[Dict[str, Any]] = []
        standby_overflow: List[Dict[str, Any]] = []
        maintenance_queue_overflow = 0

        for entry in unassigned_after_muttom:
            train_id = entry.get("trainset_id")
            dec = decision_map.get(train_id, {})
            decision = dec.get("decision", entry.get("decision"))
            first_departure = dec.get("first_departure_station") or train_map.get(train_id, {}).get("first_departure_station")
            if decision == "MAINTENANCE":
                maintenance_queue_overflow += 1
            elif decision == "SERVICE":
                service_overflow.append({**entry, "first_departure_station": first_departure, "score": dec.get("score", 0)})
            elif decision == "STANDBY":
                standby_overflow.append({**entry, "priority": dec.get("priority", 0)})

        # SERVICE allocation based on departure terminal preference
        for terminal_name, layout in self.terminal_layouts.items():
            capacity = layout["service_stabling_capacity"]
            candidates = [
                s
                for s in service_overflow
                if (s.get("first_departure_station") or "").lower().startswith(terminal_name.split()[0].lower())
            ]
            candidates.sort(key=lambda t: -t.get("score", 0))
            for idx, train in enumerate(candidates):
                if idx >= capacity:
                    continue
                terminals[terminal_name]["service_trains"].append(train["trainset_id"])
                terminals[terminal_name]["service_used"] += 1
                service_overflow.remove(train)

        # Fallback: fill remaining service overflow into any terminal with spare capacity
        for terminal_name, layout in self.terminal_layouts.items():
            remaining_cap = layout["service_stabling_capacity"] - terminals[terminal_name]["service_used"]
            if remaining_cap <= 0:
                continue
            take = service_overflow[:remaining_cap]
            service_overflow = service_overflow[remaining_cap:]
            for train in take:
                terminals[terminal_name]["service_trains"].append(train["trainset_id"])
                terminals[terminal_name]["service_used"] += 1

        # STANDBY allocation (keep min buffer near Muttom)
        remaining_standby = standby_overflow[self.min_muttom_standby_buffer :] if len(standby_overflow) > self.min_muttom_standby_buffer else []
        remaining_standby.sort(key=lambda t: -t.get("priority", 0))
        for terminal_name, layout in self.terminal_layouts.items():
            cap = layout["standby_stabling_capacity"]
            take = remaining_standby[:cap]
            remaining_standby = remaining_standby[cap:]
            for train in take:
                terminals[terminal_name]["standby_trains"].append(train["trainset_id"])
                terminals[terminal_name]["standby_used"] += 1

        unassigned_after_terminals = len(service_overflow) + len(remaining_standby)

        return terminals, {
            "unassigned_after_terminals": unassigned_after_terminals,
            "maintenance_queue_overflow": maintenance_queue_overflow,
        }

    def _build_optimized_layout(self, assignments: Dict[int, Dict[str, Any]]) -> Dict[str, List[BayAssignment]]:
        bays = []
        layout = self.depot_layouts["Muttom Depot"]
        for bay_id in range(1, layout["total_bays"] + 1):
            info = assignments.get(bay_id)
            bays.append(
                BayAssignment(
                    bay_id=bay_id,
                    role=info["role"] if info else "EMPTY",
                    trainset_id=info.get("trainset_id") if info else None,
                    turnout_time_min=None,
                    distance_to_exit_m=None,
                    notes=None,
                )
            )
        return {"Muttom Depot": bays}

    def _compute_bay_diff(
        self, current: List[BayAssignment], optimized: List[BayAssignment]
    ) -> List[Dict[str, Any]]:
        current_map = {b.trainset_id: b for b in current if b.trainset_id}
        optimized_map = {b.trainset_id: b for b in optimized if b.trainset_id}
        train_ids = set(current_map.keys()) | set(optimized_map.keys())
        diff: List[Dict[str, Any]] = []

        for train_id in sorted(train_ids):
            before = current_map.get(train_id)
            after = optimized_map.get(train_id)
            from_bay = before.bay_id if before else None
            to_bay = after.bay_id if after else None
            from_role = before.role if before else None
            to_role = after.role if after else None

            if from_bay and to_bay and from_bay == to_bay and from_role == to_role:
                move_type = "UNCHANGED"
            elif from_bay and to_bay:
                move_type = "MOVE"
            elif from_bay and not to_bay:
                move_type = "EXIT"
            else:
                move_type = "ENTER"

            diff.append(
                {
                    "trainset_id": train_id,
                    "from_bay_id": from_bay,
                    "to_bay_id": to_bay,
                    "from_role": from_role,
                    "to_role": to_role,
                    "move_type": move_type,
                }
            )
        return diff

    def _build_shunting_schedule(
        self, bay_diff: List[Dict[str, Any]], bay_positions: Dict[int, Dict[str, int]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        operations: List[Dict[str, Any]] = []
        for entry in bay_diff:
            if entry["move_type"] not in {"MOVE", "ENTER", "EXIT"}:
                continue
            from_bay = entry.get("from_bay_id")
            to_bay = entry.get("to_bay_id")
            distance = self._estimate_distance(from_bay, to_bay, bay_positions)
            estimated_time = self._estimate_shunting_time(distance, from_bay, to_bay)
            operations.append(
                {
                    "trainset_id": entry["trainset_id"],
                    "from_bay_id": from_bay,
                    "to_bay_id": to_bay,
                    "role_before": entry.get("from_role"),
                    "role_after": entry.get("to_role"),
                    "estimated_time_min": estimated_time,
                    "complexity": self._complexity(distance),
                    "priority": self._priority(entry.get("to_role")),
                }
            )

        total_time = sum(op["estimated_time_min"] for op in operations)
        buffer_minutes = self.operational_window["minutes"] - total_time
        feasible = total_time <= self.operational_window["minutes"]

        operations.sort(key=lambda x: (x["priority"], -x["estimated_time_min"]))
        for idx, op in enumerate(operations, 1):
            op["sequence"] = idx

        summary = {
            "total_operations": len(operations),
            "total_time_min": total_time,
            "available_window_min": self.operational_window["minutes"],
            "buffer_minutes": buffer_minutes,
            "feasible": feasible,
            "warning": None
            if feasible
            else f"Shunting schedule exceeds available night window by {abs(buffer_minutes)} minutes.",
        }
        return operations, summary

    def _compute_counts(
        self,
        decisions: List[Dict[str, Any]],
        optimized_layout: Dict[str, List[BayAssignment]],
        unassigned: List[Dict[str, Any]],
        terminal_allocation: Dict[str, Dict[str, Any]],
        required_service_trains: int,
    ) -> Dict[str, Dict[str, Any]]:
        decided_service = sum(1 for d in decisions if d.get("decision") == "SERVICE")
        decided_standby = sum(1 for d in decisions if d.get("decision") == "STANDBY")
        decided_maintenance = sum(1 for d in decisions if d.get("decision") == "MAINTENANCE")

        stabled_service = stabled_standby = stabled_maintenance = 0
        for bay in optimized_layout.get("Muttom Depot", []):
            if bay.role == "SERVICE":
                stabled_service += 1
            elif bay.role == "STANDBY":
                stabled_standby += 1
            elif bay.role == "MAINTENANCE":
                stabled_maintenance += 1

        terminal_service = sum(t.get("service_used", 0) for t in terminal_allocation.values())
        induction_shortfall = max(0, required_service_trains - decided_service)
        capacity_shortfall = max(0, required_service_trains - (stabled_service + terminal_service))
        effective_shortfall = max(induction_shortfall, capacity_shortfall)

        stabling_summary = {
            "stabled_service_trains": stabled_service,
            "stabled_standby_trains": stabled_standby,
            "stabled_maintenance_trains": stabled_maintenance,
            "total_stabled_trains": stabled_service + stabled_standby + stabled_maintenance,
            "unassigned_due_to_capacity": len(unassigned),
            "service_bay_capacity": 6,
            "maintenance_bay_capacity": 4,
            "standby_bay_capacity": 2,
            "total_bays": 12,
        }

        return {
            "service_requirement": {
                "required_service_trains": required_service_trains,
                "decided_service_trains": decided_service,
                "stabled_service_trains": stabled_service + terminal_service,
                "induction_shortfall": induction_shortfall,
                "capacity_shortfall": capacity_shortfall,
                "effective_service_shortfall": effective_shortfall,
            },
            "induction_summary": {
                "decided_service_trains": decided_service,
                "decided_standby_trains": decided_standby,
                "decided_maintenance_trains": decided_maintenance,
            },
            "stabling_summary": stabling_summary,
        }

    def _legacy_payload(
        self,
        optimized_layout: Dict[str, List[BayAssignment]],
        shunting_operations: List[Dict[str, Any]],
        shunting_summary: Dict[str, Any],
        decisions: List[Dict[str, Any]],
        trainsets: List[Dict[str, Any]],
        unassigned: List[Dict[str, Any]],
        terminal_allocation: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Map to legacy optimized_layout format (bay_assignments)
        bay_assignments = {
            "Muttom Depot": {
                "bay_assignments": {
                    b.trainset_id: b.bay_id for b in optimized_layout.get("Muttom Depot", []) if b.trainset_id
                },
                "shunting_operations": [
                    {
                        "trainset_id": op["trainset_id"],
                        "from_bay": op.get("from_bay_id"),
                        "to_bay": op.get("to_bay_id"),
                        "estimated_time": op["estimated_time_min"],
                        "complexity": op["complexity"],
                    }
                    for op in shunting_operations
                ],
                "total_shunting_time": shunting_summary["total_time_min"],
                "total_turnout_time": 0,
                "efficiency_score": 0,
            }
        }

        # Minimal bay_layout as expected by existing UI tiles
        bay_layout = {"Muttom Depot": optimized_layout.get("Muttom Depot", [])}

        total_positions = len(bay_assignments["Muttom Depot"]["bay_assignments"])

        return {
            "optimized_layout": bay_assignments,
            "total_optimized_positions": total_positions,
            "bay_layout": bay_layout,
            "unassigned": unassigned,
            "capacity_warning": len(unassigned) > 0,
            "total_shunting_time": shunting_summary["total_time_min"],
            "efficiency_metrics": {"overall_efficiency": 0.0, "shunting_efficiency": 0.0, "energy_savings": 0.0, "time_savings": 0.0, "depot_scores": {}},
            "induction_decisions": decisions,
            "trainsets": trainsets,
            "terminal_allocation": terminal_allocation,
        }

    def _generate_warnings(
        self, fleet_summary: FleetSummary, depot_allocations: List[DepotAllocation], unassigned_trainsets: List[Dict[str, Any]]
    ) -> List[str]:
        warnings = []
        if fleet_summary.service_shortfall > 0:
            warnings.append(
                f"Service shortfall: {fleet_summary.service_shortfall} trains short of required {fleet_summary.required_service_trains} service trains."
            )
        for depot in depot_allocations:
            if depot.capacity_warning:
                warnings.append(f"{depot.depot_name} bay capacity exceeded.")
        if unassigned_trainsets:
            warnings.append(f"{len(unassigned_trainsets)} trainset(s) could not be assigned bays due to capacity constraints.")
        return warnings

    def _empty_bay(self, bay_id: int) -> BayAssignment:
        return BayAssignment(
            bay_id=bay_id,
            role="EMPTY",
            trainset_id=None,
            turnout_time_min=None,
            distance_to_exit_m=None,
            notes=None,
        )

    def _role_from_decision(self, decision: Optional[str]) -> str:
        if decision in ("INDUCT", "SERVICE"):
            return "SERVICE"
        if decision == "MAINTENANCE":
            return "MAINTENANCE"
        return "STANDBY"
    
    def _extract_bay_number(self, bay_string: str) -> Optional[int]:
        if not bay_string:
            return None
        bay_string = str(bay_string).strip()
        patterns = [
            r"_BAY_(\d+)",
            r"Bay\s+(\d+)",
            r"B-(\d+)",
            r"B(\d+)",
            r"bay_(\d+)",
            r"^(\d+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, bay_string, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        logger.warning("Could not parse bay number from string: '%s', returning None", bay_string)
        return None
    
    def _estimate_distance(
        self, from_bay: Optional[int], to_bay: Optional[int], bay_positions: Dict[int, Dict[str, int]]
    ) -> float:
        if from_bay is None or to_bay is None:
            return 30.0  # enter/exit average shunt
        from_pos = bay_positions.get(from_bay, {"x": 0, "y": 0})
        to_pos = bay_positions.get(to_bay, {"x": 0, "y": 0})
        return math.sqrt((to_pos["x"] - from_pos["x"]) ** 2 + (to_pos["y"] - from_pos["y"]) ** 2)

    def _estimate_shunting_time(self, distance: float, from_bay: Optional[int], to_bay: Optional[int]) -> int:
        base = 6
        distance_factor = distance * 0.12
        complexity_factor = 2 if (from_bay and to_bay and abs(from_bay - to_bay) > 3) else 1
        return int(round(base + distance_factor + complexity_factor))

    def _complexity(self, distance: float) -> str:
        if distance > 120:
            return "HIGH"
        if distance > 60:
            return "MEDIUM"
        return "LOW"

    def _priority(self, role_after: Optional[str]) -> int:
        if role_after == "SERVICE":
            return 1
        if role_after == "MAINTENANCE":
            return 2
        return 3

    def _compute_turnout_time(self, bay_id: Optional[int]) -> int:
        if bay_id is None:
            return 0
        pos = self.depot_layouts["Muttom Depot"]["bay_positions"].get(bay_id, {"x": 0, "y": 0})
        distance = math.sqrt(pos["x"] ** 2 + pos["y"] ** 2)
        return int(round(6 + distance * 0.08))

    def _build_service_rollout(
        self,
        decisions: List[Dict[str, Any]],
        optimized_layout: Dict[str, List[BayAssignment]],
        terminal_allocation: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        decision_map = {d["trainset_id"]: d for d in decisions if d.get("trainset_id")}
        rollout: List[Dict[str, Any]] = []
        # Muttom bays
        for bay in optimized_layout.get("Muttom Depot", []):
            if bay.role != "SERVICE" or not bay.trainset_id:
                continue
            dec = decision_map.get(bay.trainset_id, {})
            rollout.append(
                {
                    "trainset_id": bay.trainset_id,
                    "start_location": "Muttom Depot",
                    "start_type": "DEPOT_BAY",
                    "bay_id": bay.bay_id,
                    "first_departure_station": dec.get("first_departure_station"),
                    "rollout_turnout_time_min": self._compute_turnout_time(bay.bay_id),
                }
            )

        # Terminal service rakes
        for terminal_name, alloc in terminal_allocation.items():
            exit_time = self.terminal_layouts.get(terminal_name, {}).get("terminal_exit_time_min", 7)
            for tid in alloc.get("service_trains", []):
                dec = decision_map.get(tid, {})
                rollout.append(
                    {
                        "trainset_id": tid,
                        "start_location": terminal_name,
                        "start_type": "TERMINAL_YARD",
                        "bay_id": None,
                        "first_departure_station": dec.get("first_departure_station"),
                        "rollout_turnout_time_min": exit_time,
                    }
                )
        return rollout

