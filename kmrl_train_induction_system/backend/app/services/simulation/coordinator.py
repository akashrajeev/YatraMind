"""
Simulation coordinator for multi-depot operations
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.depot import DepotConfig, DepotSimulationResult, SimulationResult
from app.services.simulation.depot_simulator import simulate_depot
from app.services.simulation.transfer_planner import plan_transfers


def run_simulation(
    depots: List[DepotConfig],
    fleet_count: int,
    service_requirement: Optional[int] = None,
    seed: Optional[int] = None,
    sim_days: int = 1,
    ai_mode: bool = True,
    train_features: Optional[Dict[str, Any]] = None
) -> SimulationResult:
    _validate_simulation_inputs(depots, fleet_count, service_requirement)

    service_requirement_auto_computed = False
    if service_requirement is None:
        service_requirement = _compute_service_requirement(fleet_count, depots)
        service_requirement_auto_computed = True

    depot_assignments = _partition_fleet(depots, fleet_count, seed)
    depot_configs = {depot.depot_id: depot for depot in depots}
    per_depot_results: Dict[str, DepotSimulationResult] = {}

    for depot in depots:
        assigned_trains = depot_assignments.get(depot.depot_id, [])
        required_service = _compute_depot_service_requirement(depot, service_requirement, depots)
        result = simulate_depot(
            depot=depot,
            assigned_trains=assigned_trains,
            required_service_n=required_service,
            seed=seed,
            ai_mode=ai_mode,
            train_features=train_features
        )
        per_depot_results[depot.depot_id] = result

    has_terminals = any(d.location_type.value == "TERMINAL_YARD" for d in depots)
    transfer_recommendations = plan_transfers(
        list(per_depot_results.values()), depot_configs, service_requirement
    ) if len(depots) > 1 or has_terminals else []

    global_summary = _compute_global_summary(
        per_depot_results, transfer_recommendations, service_requirement, depots, fleet_count
    )

    warnings: List[str] = []
    if service_requirement_auto_computed:
        warnings.append(f"required_service auto-computed: defaulted to {service_requirement}")
    total_capacity = sum(
        depot.total_bays or (depot.service_bays + depot.maintenance_bays + depot.standby_bays)
        for depot in depots
    )
    if fleet_count > total_capacity:
        warnings.append(
            f"capacity exceeded: fleet_count={fleet_count}, total_capacity={total_capacity}"
        )
    for result in per_depot_results.values():
        warnings.extend(result.warnings)
        if result.violations:
            warnings.extend([f"{result.depot_name}: {v}" for v in result.violations])

    run_id = _generate_run_id(depots, fleet_count, seed)
    config_snapshot = {
        "depots": [depot.model_dump() for depot in depots],
        "fleet_count": fleet_count,
        "service_requirement": service_requirement,
        "seed": seed,
        "sim_days": sim_days,
        "ai_mode": ai_mode
    }

    return SimulationResult(
        run_id=run_id,
        seed=seed,
        config_snapshot=config_snapshot,
        per_depot=per_depot_results,
        inter_depot_transfers=transfer_recommendations,
        global_summary=global_summary,
        warnings=warnings,
        created_at=datetime.utcnow().isoformat()
    )


def _validate_simulation_inputs(depots: List[DepotConfig], fleet_count: int, service_requirement: Optional[int]) -> None:
    if not depots:
        raise ValueError("At least one depot required")
    if fleet_count <= 0:
        raise ValueError("Fleet count must be positive")
    if service_requirement is not None and service_requirement < 0:
        raise ValueError("Service requirement must be non-negative")


def _compute_service_requirement(fleet_count: int, depots: List[DepotConfig]) -> int:
    if fleet_count <= 0:
        return 13
    return max(13, int(fleet_count * 0.3))


def _compute_depot_service_requirement(depot: DepotConfig, global_requirement: int, all_depots: List[DepotConfig]) -> int:
    total_service_bays = sum(d.service_bays for d in all_depots)
    if total_service_bays == 0:
        return 0
    return int(global_requirement * (depot.service_bays / total_service_bays))


def _partition_fleet(depots: List[DepotConfig], fleet_count: int, seed: Optional[int] = None) -> Dict[str, List[str]]:
    """Partition the complete fleet across depots; capacity is reported separately as an operational warning."""
    train_ids = [f"TRAIN_{i+1:03d}" for i in range(fleet_count)]
    assignments: Dict[str, List[str]] = {depot.depot_id: [] for depot in depots}
    if not depots:
        return assignments

    # Stable weighted round-robin: keeps deterministic output while never dropping overflow trains.
    weights = [max(1, depot.total_bays or (depot.service_bays + depot.maintenance_bays + depot.standby_bays)) for depot in depots]
    total_weight = sum(weights)
    for index, train_id in enumerate(train_ids):
        cursor = index % total_weight
        for depot, weight in zip(depots, weights):
            if cursor < weight:
                assignments[depot.depot_id].append(train_id)
                break
            cursor -= weight
    return assignments


def _compute_global_summary(
    per_depot: Dict[str, DepotSimulationResult],
    transfers: list,
    required_service: int,
    depots: List[DepotConfig],
    fleet_count: int
) -> Dict[str, Any]:
    total_shunting_time = int(sum(r.shunting_summary.get("total_time_min", 0) for r in per_depot.values()))
    total_turnout_time = int(sum(r.shunting_summary.get("total_time_min", 0) * 0.8 for r in per_depot.values()))
    total_service = int(sum(r.stabling_summary.get("service_trains", 0) for r in per_depot.values()))
    total_capacity = int(sum(d.total_bays or (d.service_bays + d.maintenance_bays + d.standby_bays) for d in depots))
    return {
        "service_trains": total_service,
        "required_service": required_service,
        "stabled_service": total_service,
        "service_shortfall": max(0, required_service - total_service),
        "shunting_time": total_shunting_time,
        "turnout_time": total_turnout_time,
        "total_capacity": total_capacity,
        "fleet": fleet_count,
        "transfers_recommended": len([t for t in transfers if t.recommended]),
        "shunting_feasible": all(r.shunting_summary.get("feasible", False) for r in per_depot.values()),
        "estimated_energy_savings_kwh": total_service * 100,
        "total_transfer_cost": sum(t.cost_estimate for t in transfers if t.recommended),
    }


def _generate_run_id(depots: List[DepotConfig], fleet_count: int, seed: Optional[int]) -> str:
    config_str = json.dumps({"depots": [d.depot_id for d in depots], "fleet": fleet_count, "seed": seed}, sort_keys=True)
    return f"SIM_{hashlib.md5(config_str.encode()).hexdigest()[:12]}"
