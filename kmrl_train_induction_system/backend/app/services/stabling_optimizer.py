"""Compatibility wrapper around the stable legacy stabling implementation.

The large operational algorithm remains isolated in ``stabling_optimizer_legacy``
while this adapter exposes the legacy helper methods and enriches the rich
response with the fields required by existing callers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .stabling_optimizer_legacy import StablingGeometryOptimizer as _LegacyStablingGeometryOptimizer


class StablingGeometryOptimizer(_LegacyStablingGeometryOptimizer):
    """Backward-compatible adapter over the legacy stabling optimizer."""

    def _assign_role_bays(
        self,
        trains: List[Dict[str, Any]],
        bays: List[int],
        used_bays: set[int],
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
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

    def _assign_service_bays(
        self,
        service_trains: List[Dict[str, Any]],
        depot_layout: Dict[str, Any],
        used_bays: set[int],
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        return self._assign_role_bays(service_trains, depot_layout.get("service_bays", []), used_bays)

    def _assign_maintenance_bays(
        self,
        maintenance_trains: List[Dict[str, Any]],
        depot_layout: Dict[str, Any],
        used_bays: set[int],
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        return self._assign_role_bays(maintenance_trains, depot_layout.get("maintenance_bays", []), used_bays)

    def _assign_standby_bays(
        self,
        standby_trains: List[Dict[str, Any]],
        depot_layout: Dict[str, Any],
        used_bays: set[int],
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        all_bays = list(range(1, int(depot_layout.get("total_bays", 0)) + 1))
        return self._assign_role_bays(standby_trains, all_bays, used_bays)

    async def _optimize_depot_layout(
        self,
        depot_name: str,
        trainsets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        layout_name = "Muttom Depot" if depot_name == "Muttom" else depot_name
        layout = self.depot_layouts.get(layout_name)
        if not layout:
            layout = {
                "total_bays": 12,
                "maintenance_bays": [1, 2, 3, 4],
                "standby_bays": [5, 6],
                "service_bays": [7, 8, 9, 10, 11, 12],
            }

        service = [
            t for t in trainsets
            if str((t.get("induction_decision") or {}).get("decision", "")).upper() in {"INDUCT", "SERVICE"}
        ]
        maintenance = [
            t for t in trainsets
            if str((t.get("induction_decision") or {}).get("decision", "")).upper() == "MAINTENANCE"
        ]
        standby = [t for t in trainsets if t not in service and t not in maintenance]

        used_bays: set[int] = set()
        assignments: Dict[str, int] = {}
        unassigned: List[Dict[str, Any]] = []
        for trains, bays in (
            (service, layout.get("service_bays", [])),
            (maintenance, layout.get("maintenance_bays", [])),
            (standby, layout.get("standby_bays", [])),
        ):
            result, overflow = self._assign_role_bays(trains, bays, used_bays)
            assignments.update(result)
            used_bays.update(result.values())
            unassigned.extend(overflow)
        return {"bay_assignments": assignments, "unassigned": unassigned}

    async def optimize_stabling_geometry(
        self,
        trainsets: List[Dict[str, Any]],
        induction_decisions: List[Dict[str, Any]],
        fleet_req: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = await super().optimize_stabling_geometry(trainsets, induction_decisions, fleet_req)

        layout = self.depot_layouts["Muttom Depot"]
        optimized_bays = result.get("optimized_bay_layout", {}).get("Muttom Depot", [])
        assigned = sum(1 for bay in optimized_bays if getattr(bay, "trainset_id", None))
        unassigned = list(result.get("unassigned_trainsets", []))
        capacity = result.get("capacity_summary") or {
            "total_assigned": assigned,
            "total_capacity": int(layout.get("total_bays", 12)),
            "unassigned_due_to_capacity": len(unassigned),
        }
        result["capacity_summary"] = capacity

        maintenance_queue = result.get("maintenance_queue") or [
            {"trainset_id": item.get("trainset_id"), "reason": "No bay capacity"}
            for item in unassigned
            if str(item.get("decision", "")).upper() in {"MAINTENANCE", "MAINT"}
        ]
        result["maintenance_queue"] = maintenance_queue

        shunting_summary = result.get("shunting_summary", {})
        required_minutes = int(shunting_summary.get("total_time_min", 0) or 0)
        available_minutes = int(self.operational_window.get("minutes", 120))
        result["shunting_window"] = {
            "available_minutes": available_minutes,
            "required_minutes": required_minutes,
            "buffer_minutes": max(0, available_minutes - required_minutes),
            "feasible": required_minutes <= available_minutes,
        }
        return result

    async def generate_rich_stabling_geometry(
        self,
        trainsets: List[Dict[str, Any]],
        induction_decisions: List[Dict[str, Any]],
        fleet_req: Optional[Dict[str, Any]] = None,
    ):
        rich = await super().generate_rich_stabling_geometry(trainsets, induction_decisions, fleet_req)
        result = await self.optimize_stabling_geometry(trainsets, induction_decisions, fleet_req)
        return rich.model_copy(update={
            "shunting_operations": result.get("shunting_operations", []),
            "capacity_summary": result.get("capacity_summary"),
            "unassigned_trainsets": result.get("unassigned_trainsets", []),
            "maintenance_queue": result.get("maintenance_queue", []),
            "shunting_window": result.get("shunting_window"),
            "service_requirement": result.get("service_requirement"),
            "induction_summary": result.get("induction_summary"),
            "stabling_summary": result.get("stabling_summary"),
        })


__all__ = ["StablingGeometryOptimizer"]
