"""
Simulation coordinator for multi-depot operations
"""
import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.depot import DepotConfig, DepotSimulationResult, SimulationResult, TransferRecommendation
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
    """
    Run multi-depot simulation
    
    Args:
        depots: List of depot configurations
        fleet_count: Total number of trains in fleet
        service_requirement: Required service trains (auto-computed if None)
        seed: Random seed for reproducibility
        sim_days: Number of simulation days
        ai_mode: Whether to use AI/ML services
        train_features: Optional train feature dictionary
    
    Returns:
        SimulationResult with per-depot results and global summary
    """
    # Validate inputs
    _validate_simulation_inputs(depots, fleet_count, service_requirement)
    
    # Compute service requirement if not provided
    service_requirement_auto_computed = False
    if service_requirement is None:
        service_requirement = _compute_service_requirement(fleet_count, depots)
        service_requirement_auto_computed = True
    
    # Partition fleet into depot assignments
    depot_assignments = _partition_fleet(depots, fleet_count, seed)
    
    # Run per-depot simulations
    depot_configs = {depot.depot_id: depot for depot in depots}
    per_depot_results: Dict[str, DepotSimulationResult] = {}
    
    for depot in depots:
        assigned_trains = depot_assignments.get(depot.depot_id, [])
        required_service = _compute_depot_service_requirement(
            depot, service_requirement, depots
        )
        
        result = simulate_depot(
            depot=depot,
            assigned_trains=assigned_trains,
            required_service_n=required_service,
            seed=seed,
            ai_mode=ai_mode,
            train_features=train_features
        )
        
        per_depot_results[depot.depot_id] = result
    
    # Plan inter-depot transfers (only if multiple depots or terminals present)
    has_terminals = any(d.location_type.value == "TERMINAL_YARD" for d in depots)
    if len(depots) > 1 or has_terminals:
        transfer_recommendations = plan_transfers(
            list(per_depot_results.values()),
            depot_configs,
            service_requirement
        )
    else:
        transfer_recommendations = []
    
    # Compute global summary
    global_summary = _compute_global_summary(
        per_depot_results, transfer_recommendations, service_requirement, depots, fleet_count
    )
    
    # Collect warnings
    warnings = []
    if service_requirement_auto_computed:
        warnings.append(f"required_service auto-computed: defaulted to {service_requirement}")
    for result in per_depot_results.values():
        warnings.extend(result.warnings)
        if result.violations:
            warnings.extend([f"{result.depot_name}: {v}" for v in result.violations])
    
    # Generate run ID
    run_id = _generate_run_id(depots, fleet_count, seed)
    
    # Create config snapshot
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


def _validate_simulation_inputs(
    depots: List[DepotConfig],
    fleet_count: int,
    service_requirement: Optional[int]
) -> None:
    """Validate simulation inputs"""
    if not depots:
        raise ValueError("At least one depot required")
    
    if fleet_count <= 0:
        raise ValueError("Fleet count must be positive")
    
    total_capacity = sum(depot.total_bays or 0 for depot in depots)
    if fleet_count > total_capacity * 1.2:  # Allow 20% overcapacity warning
        # This will be a warning, not an error
        pass
    
    if service_requirement is not None and service_requirement < 0:
        raise ValueError("Service requirement must be non-negative")


def _compute_service_requirement(
    fleet_count: int,
    depots: List[DepotConfig]
) -> int:
    """
    Compute service requirement from fleet size
    Default to 13 if not computable, otherwise use heuristic
    """
    if fleet_count <= 0:
        return 13  # Default fallback
    computed = int(fleet_count * 0.3)
    return max(13, computed)  # Minimum 13


def _compute_depot_service_requirement(
    depot: DepotConfig,
    global_requirement: int,
    all_depots: List[DepotConfig]
) -> int:
    """
    Compute service requirement for a specific depot
    Proportional to depot's service bay capacity
    """
    total_service_bays = sum(d.depot_id != depot.depot_id and d.service_bays or 0 for d in all_depots)
    total_service_bays += depot.service_bays
    
    if total_service_bays == 0:
        return 0
    
    return int(global_requirement * (depot.service_bays / total_service_bays))


def _partition_fleet(
    depots: List[DepotConfig],
    fleet_count: int,
    seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Partition fleet across depots
    Uses round-robin assignment weighted by depot capacity
    Distributes overflow to terminals if available
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    # Generate train IDs
    train_ids = [f"TRAIN_{i+1:03d}" for i in range(fleet_count)]
    
    # Separate terminals from full depots
    full_depots = [d for d in depots if d.location_type.value != "TERMINAL_YARD"]
    terminals = [d for d in depots if d.location_type.value == "TERMINAL_YARD"]
    
    # Weight full depots by capacity
    depot_weights = []
    for depot in full_depots:
        weight = depot.total_bays or 1
        depot_weights.append((depot.depot_id, weight))
    
    # Distribute trains to full depots first
    assignments: Dict[str, List[str]] = {depot.depot_id: [] for depot in depots}
    
    total_capacity = sum(depot.total_bays or 0 for depot in full_depots)
    train_idx = 0
    
    # Assign to full depots up to capacity
    while train_idx < len(train_ids) and train_idx < total_capacity:
        for depot_id, weight in depot_weights:
            if train_idx >= len(train_ids) or train_idx >= total_capacity:
                break
            # Assign trains proportional to weight
            num_to_assign = max(1, int(weight / sum(w for _, w in depot_weights) * min(total_capacity, len(train_ids))))
            assignments[depot_id].extend(
                train_ids[train_idx:train_idx + num_to_assign]
            )
            train_idx += num_to_assign
            if train_idx >= len(train_ids) or train_idx >= total_capacity:
                break
    
    # Distribute overflow to terminals
    overflow = len(train_ids) - train_idx
    if overflow > 0 and terminals:
        terminal_idx = 0
        for terminal in terminals:
            if train_idx >= len(train_ids):
                break
            terminal_capacity = terminal.standby_bays
            num_to_assign = min(terminal_capacity, overflow)
            assignments[terminal.depot_id].extend(
                train_ids[train_idx:train_idx + num_to_assign]
            )
            train_idx += num_to_assign
            overflow -= num_to_assign
            if train_idx >= len(train_ids):
                break
    
    return assignments


def _compute_global_summary(
    per_depot: Dict[str, DepotSimulationResult],
    transfers: List[TransferRecommendation],
    required_service: int,
    depots: List[DepotConfig],
    fleet_count: int
) -> Dict[str, Any]:
    """Compute global KPIs with aligned schema for UI"""
    # Aggregate shunting time from all depots
    total_shunting_time = int(sum(
        result.shunting_summary.get("total_time_min", 0)
        for result in per_depot.values()
    ))
    
    # Aggregate turnout time (80% of shunting time as estimate)
    total_turnout_time = int(sum(
        result.shunting_summary.get("total_time_min", 0) * 0.8
        for result in per_depot.values()
    ))
    
    # Aggregate service trains
    total_service = int(sum(
        result.stabling_summary.get("service_trains", 0)
        for result in per_depot.values()
    ))
    
    # Compute stabled service (same as service_trains for now)
    stabled_service = total_service
    
    # Compute service shortfall
    service_shortfall = max(0, required_service - total_service)
    
    # Compute total capacity across all depots
    total_capacity = int(sum(
        depot.total_bays or (depot.service_bays + depot.maintenance_bays + depot.standby_bays)
        for depot in depots
    ))
    
    # Check shunting feasibility (all depots must be feasible)
    shunting_feasible = all(
        result.shunting_summary.get("feasible", False)
        for result in per_depot.values()
    )
    
    # Count recommended transfers
    transfers_recommended = len([t for t in transfers if t.recommended])
    
    # Estimate energy savings (simplified)
    estimated_energy_savings = total_service * 100  # kWh per train
    
    return {
        "service_trains": total_service,
        "required_service": required_service,
        "stabled_service": stabled_service,
        "service_shortfall": service_shortfall,
        "shunting_time": total_shunting_time,
        "turnout_time": total_turnout_time,
        "total_capacity": total_capacity,
        "fleet": fleet_count,
        "transfers_recommended": transfers_recommended,
        "shunting_feasible": shunting_feasible,
        "estimated_energy_savings_kwh": estimated_energy_savings,
        "total_transfer_cost": sum(t.cost_estimate for t in transfers if t.recommended)
    }


def _generate_run_id(
    depots: List[DepotConfig],
    fleet_count: int,
    seed: Optional[int]
) -> str:
    """Generate deterministic run ID"""
    config_str = json.dumps({
        "depots": [d.depot_id for d in depots],
        "fleet": fleet_count,
        "seed": seed
    }, sort_keys=True)
    
    hash_obj = hashlib.md5(config_str.encode())
    return f"SIM_{hash_obj.hexdigest()[:12]}"

