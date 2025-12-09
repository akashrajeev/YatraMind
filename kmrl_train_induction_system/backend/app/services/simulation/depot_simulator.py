"""
Depot-level simulation for stabling, shunting, and bay allocation
"""
import random
from typing import Any, Dict, List, Optional

from app.models.depot import DepotConfig, DepotSimulationResult


def simulate_depot(
    depot: DepotConfig,
    assigned_trains: List[str],
    required_service_n: int,
    seed: Optional[int] = None,
    ai_mode: bool = True,
    train_features: Optional[Dict[str, Any]] = None
) -> DepotSimulationResult:
    """
    Simulate a single depot's operations
    
    Args:
        depot: Depot configuration
        assigned_trains: List of train IDs assigned to this depot
        required_service_n: Number of trains required for service
        seed: Random seed for reproducibility
        ai_mode: Whether to use AI/ML services (if available)
        train_features: Optional train feature dictionary for AI mode
    
    Returns:
        DepotSimulationResult with bay layouts, shunting ops, and KPIs
    """
    if seed is not None:
        random.seed(seed)
    
    # Initialize bay layouts
    bay_layout_before = _initialize_bay_layout(depot, assigned_trains)
    
    # Run optimization (AI or heuristic)
    if ai_mode:
        bay_layout_after, shunting_ops = _ai_optimize_depot(
            depot, assigned_trains, required_service_n, train_features
        )
    else:
        bay_layout_after, shunting_ops = _heuristic_optimize_depot(
            depot, assigned_trains, required_service_n
        )
    
    # Compute bay differences
    bay_diff = _compute_bay_diff(bay_layout_before, bay_layout_after)
    
    # Compute shunting summary
    shunting_summary = _compute_shunting_summary(shunting_ops, depot)
    
    # Compute stabling summary
    stabling_summary = _compute_stabling_summary(
        bay_layout_after, depot, required_service_n
    )
    
    # Compute KPIs
    kpis = _compute_depot_kpis(
        depot, stabling_summary, shunting_summary, required_service_n
    )
    
    # Check for violations and warnings
    warnings, violations = _check_constraints(
        depot, bay_layout_after, stabling_summary, shunting_summary
    )
    
    return DepotSimulationResult(
        depot_id=depot.depot_id,
        depot_name=depot.name,
        assigned_trains=assigned_trains,
        stabling_summary=stabling_summary,
        bay_layout_before=bay_layout_before,
        bay_layout_after=bay_layout_after,
        bay_diff=bay_diff,
        shunting_operations=shunting_ops,
        shunting_summary=shunting_summary,
        kpis=kpis,
        warnings=warnings,
        violations=violations
    )


def _initialize_bay_layout(depot: DepotConfig, trains: List[str]) -> Dict[str, Any]:
    """Initialize bay layout with current train positions"""
    layout = {
        "service": {},
        "maintenance": {},
        "standby": {}
    }
    
    # Distribute trains across bays (simple round-robin)
    bay_idx = 0
    for train_id in trains:
        if bay_idx < depot.service_bays:
            layout["service"][f"BAY_{bay_idx + 1}"] = train_id
        elif bay_idx < depot.service_bays + depot.maintenance_bays:
            layout["maintenance"][f"BAY_{bay_idx + 1}"] = train_id
        else:
            layout["standby"][f"BAY_{bay_idx + 1}"] = train_id
        bay_idx += 1
    
    return layout


def _ai_optimize_depot(
    depot: DepotConfig,
    trains: List[str],
    required_service: int,
    train_features: Optional[Dict[str, Any]] = None
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Use AI/ML services to optimize depot layout
    Falls back to heuristic if AI services unavailable
    """
    # TODO: Integrate with actual AI services
    # For now, use heuristic as fallback
    return _heuristic_optimize_depot(depot, trains, required_service)


def _heuristic_optimize_depot(
    depot: DepotConfig,
    trains: List[str],
    required_service: int
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Heuristic optimization: assign highest priority trains to service bays
    For terminals, only use standby bays
    """
    # For terminals, only use standby bays
    if depot.location_type.value == "TERMINAL_YARD":
        standby_trains = trains[:min(depot.standby_bays, len(trains))]
        layout = {
            "service": {},
            "maintenance": {},
            "standby": {f"BAY_{i+1}": train_id for i, train_id in enumerate(standby_trains)}
        }
        shunting_ops = []
        return layout, shunting_ops
    
    # For full depots, assign to service/maintenance/standby
    service_trains = trains[:min(required_service, len(trains), depot.service_bays)]
    remaining = trains[len(service_trains):]
    
    # Assign maintenance (next M trains)
    maintenance_trains = remaining[:min(depot.maintenance_bays, len(remaining))]
    standby_trains = remaining[len(maintenance_trains):]
    
    # Build optimized layout
    layout = {
        "service": {f"BAY_{i+1}": train_id for i, train_id in enumerate(service_trains)},
        "maintenance": {f"BAY_{i+1}": train_id for i, train_id in enumerate(maintenance_trains)},
        "standby": {f"BAY_{i+1}": train_id for i, train_id in enumerate(standby_trains)}
    }
    
    # Generate shunting operations (simplified: estimate 5 min per move)
    shunting_ops = []
    num_moves = len(service_trains) + len(maintenance_trains) + len(standby_trains)
    for i in range(num_moves):
        shunting_ops.append({
            "operation_id": f"SHUNT_{i+1}",
            "estimated_time_min": 5,
            "type": "move"
        })
    
    return layout, shunting_ops


def _compute_bay_diff(
    before: Dict[str, Any],
    after: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compute differences between before and after layouts"""
    diff = []
    
    for bay_type in ["service", "maintenance", "standby"]:
        before_bays = before.get(bay_type, {})
        after_bays = after.get(bay_type, {})
        
        # Find moves
        for bay_id, train_id in after_bays.items():
            if bay_id not in before_bays or before_bays[bay_id] != train_id:
                # Find where train was before
                old_bay = None
                for old_type in ["service", "maintenance", "standby"]:
                    if bay_id in before.get(old_type, {}):
                        old_bay = (old_type, bay_id)
                        break
                
                diff.append({
                    "train_id": train_id,
                    "from": old_bay,
                    "to": (bay_type, bay_id),
                    "type": "move"
                })
    
    return diff


def _compute_shunting_summary(
    shunting_ops: List[Dict[str, Any]],
    depot: DepotConfig
) -> Dict[str, Any]:
    """Compute shunting operation summary"""
    total_time = sum(op.get("estimated_time_min", 0) for op in shunting_ops)
    
    return {
        "total_operations": len(shunting_ops),
        "total_time_min": total_time,
        "feasible": total_time <= depot.max_shunting_window_min,
        "window_min": depot.max_shunting_window_min
    }


def _compute_stabling_summary(
    layout: Dict[str, Any],
    depot: DepotConfig,
    required_service: int
) -> Dict[str, Any]:
    """Compute stabling summary"""
    service_count = len(layout.get("service", {}))
    maintenance_count = len(layout.get("maintenance", {}))
    standby_count = len(layout.get("standby", {}))
    
    service_shortfall = max(0, required_service - service_count)
    capacity_shortfall = max(0, (service_count + maintenance_count + standby_count) - depot.total_bays)
    
    return {
        "service_trains": service_count,
        "maintenance_trains": maintenance_count,
        "standby_trains": standby_count,
        "total_stabled": service_count + maintenance_count + standby_count,
        "service_shortfall": service_shortfall,
        "capacity_shortfall": capacity_shortfall,
        "service_capacity": depot.service_bays,
        "maintenance_capacity": depot.maintenance_bays,
        "standby_capacity": depot.standby_bays,
        "total_capacity": depot.total_bays
    }


def _compute_depot_kpis(
    depot: DepotConfig,
    stabling_summary: Dict[str, Any],
    shunting_summary: Dict[str, Any],
    required_service: int
) -> Dict[str, Any]:
    """Compute depot-level KPIs"""
    service_utilization = (
        stabling_summary["service_trains"] / depot.service_bays
        if depot.service_bays > 0 else 0.0
    )
    
    return {
        "service_utilization": service_utilization,
        "capacity_utilization": (
            stabling_summary["total_stabled"] / depot.total_bays
            if depot.total_bays > 0 else 0.0
        ),
        "service_shortfall": stabling_summary["service_shortfall"],
        "shunting_time_min": shunting_summary["total_time_min"],
        "shunting_feasible": shunting_summary["feasible"]
    }


def _check_constraints(
    depot: DepotConfig,
    layout: Dict[str, Any],
    stabling_summary: Dict[str, Any],
    shunting_summary: Dict[str, Any]
) -> tuple[List[str], List[str]]:
    """Check for constraint violations and warnings"""
    warnings = []
    violations = []
    
    # Capacity warnings
    if stabling_summary["capacity_shortfall"] > 0:
        violations.append(
            f"Capacity exceeded: {stabling_summary['capacity_shortfall']} trains over capacity"
        )
    
    # Shunting window violations
    if not shunting_summary["feasible"]:
        violations.append(
            f"Shunting window exceeded: {shunting_summary['total_time_min']} min > {shunting_summary['window_min']} min"
        )
    
    # Service shortfall warnings
    if stabling_summary["service_shortfall"] > 0:
        warnings.append(
            f"Service shortfall: {stabling_summary['service_shortfall']} trains needed"
        )
    
    return warnings, violations

