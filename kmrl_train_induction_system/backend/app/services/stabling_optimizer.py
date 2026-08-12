"""Compatibility wrapper around the stable legacy stabling implementation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.models.trainset import BayAssignment
from .stabling_optimizer_legacy import StablingGeometryOptimizer as _LegacyStablingGeometryOptimizer


class StablingGeometryOptimizer(_LegacyStablingGeometryOptimizer):
    """Backward-compatible adapter over the legacy stabling optimizer."""

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
        all_bays = list(range(1, int(depot_layout.get("total_bays", 0)) + 1))
        return self._assign_role_bays(standby_trains, all_bays, used_bays)

    async def _optimize_depot_layout(self, depot_name: str, trainsets: List[Dict[str, Any]]) -> Dict[str, Any]:
        layout_name = "Muttom Depot" if depot_name == "Muttom" else depot_name
        layout = self.depot_layouts.get(layout_name) or {"total_bays": 12, "maintenance_bays": [1, 2, 3, 4], "standby_bays": [5, 6], "service_bays": [7, 8, 9, 10, 11, 12]}
        service = [t for t in trainsets if str((t.get("induction_decision") or {}).get("decision", "")).upper() in {"INDUCT", "SERVICE"}]
        maintenance = [t for t in trainsets if str((t.get("induction_decision") or {}).get("decision", "")).upper() == "MAINTENANCE"]
        standby = [t for t in trainsets if t not in service and t not in maintenance]
        used_bays: set[int] = set()
        assignments: Dict[str, int] = {}
        unassigned: List[Dict[str, Any]] = []
        for trains, bays in ((service, layout.get("service_bays", [])), (maintenance, layout.get("maintenance_bays", [])), (standby, layout.get("standby_bays", []))):
            result, overflow = self._assign_role_bays(trains, bays, used_bays)
            assignments.update(result)
            used_bays.update(result.values())
            unassigned.extend(overflow)
        return {"bay_assignments": assignments, "unassigned": unassigned}

    @staticmethod
    def _decision_map(decisions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {str(d.get("trainset_id")): d for d in decisions if d.get("trainset_id")}

    def _enrich_dead_km(self, optimized_layout: Dict[str, List[BayAssignment]], trainsets: List[Dict[str, Any]]) -> Dict[str, List[BayAssignment]]:
        train_map = {str(t.get("trainset_id")): t for t in trainsets}
        enriched: Dict[str, List[BayAssignment]] = {}
        for depot_name, bays in optimized_layout.items():
            out: List[BayAssignment] = []
            for bay in bays:
                if not bay.trainset_id:
                    out.append(bay)
                    continue
                train = train_map.get(str(bay.trainset_id), {})
                current_depot = str((train.get("current_location") or {}).get("depot", "")).strip().lower()
                depot_key = depot_name.strip().lower()
                inbound = 0.0 if current_depot in {depot_key, "muttom", "muttom depot"} else 5.0
                outbound = 0.0
                total = inbound + outbound
                out.append(bay.model_copy(update={
                    "dead_km": {"in": inbound, "out": outbound, "total": total},
                    "dead_km_in": inbound,
                    "dead_km_out": outbound,
                    "stabled_at": depot_name,
                }))
            enriched[depot_name] = out
        return enriched

    async def optimize_stabling_geometry(self, trainsets: List[Dict[str, Any]], induction_decisions: List[Dict[str, Any]], fleet_req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = await super().optimize_stabling_geometry(trainsets, induction_decisions, fleet_req)
        layout = self.depot_layouts["Muttom Depot"]
        optimized_bays = result.get("optimized_bay_layout", {}).get("Muttom Depot", [])
        enriched_layout = self._enrich_dead_km(result.get("optimized_bay_layout", {}), trainsets)
        result["optimized_bay_layout"] = enriched_layout
        assigned = sum(1 for bay in enriched_layout.get("Muttom Depot", []) if getattr(bay, "trainset_id", None))
        unassigned = list(result.get("unassigned_trainsets", []))
        capacity = result.get("capacity_summary") or {"total_assigned": assigned, "total_capacity": int(layout.get("total_bays", 12)), "unassigned_due_to_capacity": len(unassigned)}
        result["capacity_summary"] = capacity

        decision_map = self._decision_map(induction_decisions)
        maintenance = [d for d in induction_decisions if str(d.get("decision", "")).upper() == "MAINTENANCE"]
        maintenance_capacity = int(layout.get("maintenance_bay_capacity", 4))
        maintenance_queue = result.get("maintenance_queue") or []
        if len(maintenance) > maintenance_capacity:
            maintenance_queue = [
                {"trainset_id": d.get("trainset_id"), "reason": "MAINTENANCE_BAY_CAPACITY", "severity": d.get("maintenance_severity", "LIGHT")}
                for d in maintenance[maintenance_capacity:]
            ]
        elif not maintenance_queue:
            maintenance_queue = [
                {"trainset_id": item.get("trainset_id"), "reason": "No bay capacity"}
                for item in unassigned
                if str(decision_map.get(str(item.get("trainset_id")), {}).get("decision", item.get("decision", ""))).upper() == "MAINTENANCE"
            ]
        result["maintenance_queue"] = maintenance_queue

        shunting_summary = result.get("shunting_summary", {})
        required_minutes = int(shunting_summary.get("total_time_min", 0) or 0)
        available_minutes = int(self.operational_window.get("minutes", 120))
        result["shunting_window"] = {"available_minutes": available_minutes, "required_minutes": required_minutes, "buffer_minutes": max(0, available_minutes - required_minutes), "feasible": required_minutes <= available_minutes}
        return result

    async def generate_rich_stabling_geometry(self, trainsets: List[Dict[str, Any]], induction_decisions: List[Dict[str, Any]], fleet_req: Optional[Dict[str, Any]] = None):
        result = await self.optimize_stabling_geometry(trainsets, induction_decisions, fleet_req)
        rich = await super().generate_rich_stabling_geometry(trainsets, induction_decisions, fleet_req)
        return rich.model_copy(update={
            "bay_layout": result.get("optimized_bay_layout", {}),
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
