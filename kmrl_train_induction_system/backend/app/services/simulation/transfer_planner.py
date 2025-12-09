"""
Inter-depot transfer planner for optimizing global fleet distribution
"""
from typing import Dict, List, Optional

from app.models.depot import DepotConfig, DepotSimulationResult, TransferRecommendation


def plan_transfers(
    depots_results: List[DepotSimulationResult],
    depot_configs: Dict[str, DepotConfig],
    global_required_service: int,
    transfer_budget_hours: float = 2.0
) -> List[TransferRecommendation]:
    """
    Plan inter-depot transfers to optimize global service coverage
    
    Args:
        depots_results: Results from each depot simulation
        depot_configs: Depot configuration map
        global_required_service: Total service requirement across all depots
        transfer_budget_hours: Maximum time budget for transfers
    
    Returns:
        List of transfer recommendations prioritized by benefit
    """
    recommendations = []
    
    # Compute global shortfall
    total_service = sum(
        result.stabling_summary.get("service_trains", 0)
        for result in depots_results
    )
    global_shortfall = max(0, global_required_service - total_service)
    
    if global_shortfall == 0:
        return recommendations  # No transfers needed
    
    # Find depots with excess capacity and depots with shortfall
    excess_depots = []
    shortfall_depots = []
    
    for result in depots_results:
        depot_config = depot_configs.get(result.depot_id)
        if not depot_config:
            continue
        
        service_shortfall = result.stabling_summary.get("service_shortfall", 0)
        service_capacity = depot_config.service_bays
        service_used = result.stabling_summary.get("service_trains", 0)
        available_capacity = service_capacity - service_used
        
        if available_capacity > 0 and len(result.assigned_trains) > service_used:
            excess_depots.append({
                "depot_id": result.depot_id,
                "available_capacity": available_capacity,
                "excess_trains": len(result.assigned_trains) - service_used,
                "result": result
            })
        
        if service_shortfall > 0:
            shortfall_depots.append({
                "depot_id": result.depot_id,
                "shortfall": service_shortfall,
                "result": result
            })
    
    # Generate transfer recommendations
    for shortfall_info in shortfall_depots:
        shortfall_depot_id = shortfall_info["depot_id"]
        shortfall = shortfall_info["shortfall"]
        shortfall_depot_config = depot_configs[shortfall_depot_id]
        
        for excess_info in excess_depots:
            excess_depot_id = excess_info["depot_id"]
            excess_depot_config = depot_configs[excess_depot_id]
            
            if excess_depot_id == shortfall_depot_id:
                continue  # Skip same depot
            
            # Find trains that can be transferred
            excess_result = excess_info["result"]
            transferable_trains = [
                train_id for train_id in excess_result.assigned_trains
                if train_id not in excess_result.bay_layout_after.get("service", {}).values()
            ]
            
            if not transferable_trains:
                continue
            
            # Compute transfer cost and benefit
            num_transfers = min(shortfall, excess_info["available_capacity"], len(transferable_trains))
            
            for i in range(num_transfers):
                train_id = transferable_trains[i]
                cost, benefit, dead_km, time_hours = _compute_transfer_metrics(
                    excess_depot_config,
                    shortfall_depot_config,
                    train_id
                )
                
                # Check if within budget
                if time_hours > transfer_budget_hours:
                    continue
                
                recommendation = TransferRecommendation(
                    from_depot=excess_depot_id,
                    to_depot=shortfall_depot_id,
                    train_id=train_id,
                    cost_estimate=cost,
                    benefit_estimate=benefit,
                    reason=f"Reduce service shortfall at {shortfall_depot_config.name}",
                    feasibility=True,
                    recommended=benefit > cost,
                    dead_km=dead_km,
                    estimated_time_hours=time_hours
                )
                
                recommendations.append(recommendation)
    
    # Sort by benefit/cost ratio
    recommendations.sort(key=lambda x: x.benefit_estimate / max(x.cost_estimate, 1), reverse=True)
    
    return recommendations


def _compute_transfer_metrics(
    from_depot: DepotConfig,
    to_depot: DepotConfig,
    train_id: str
) -> tuple[float, float, float, float]:
    """
    Compute transfer cost, benefit, dead km, and time
    
    Returns:
        (cost, benefit, dead_km, time_hours)
    """
    # Compute distance (simplified: use coordinates if available)
    dead_km = _estimate_distance(from_depot, to_depot)
    
    # Cost components
    dead_km_cost = dead_km * 3000  # ₹3K per km
    crew_cost = 2000  # ₹2K crew cost
    energy_cost = dead_km * 50  # ₹50 per km energy
    downtime_cost = 10000  # ₹10K estimated downtime
    
    total_cost = dead_km_cost + crew_cost + energy_cost + downtime_cost
    
    # Benefit: value of additional service train
    service_value_per_day = 50000  # ₹50K per train per day
    benefit = service_value_per_day * 0.5  # Assume 0.5 day benefit
    
    # Time estimate (simplified: 30 min base + 2 min per km)
    time_hours = (30 + dead_km * 2) / 60
    
    return total_cost, benefit, dead_km, time_hours


def _estimate_distance(depot1: DepotConfig, depot2: DepotConfig) -> float:
    """Estimate distance between two depots"""
    if depot1.coordinates and depot2.coordinates:
        # Simple Euclidean distance (not accurate for real-world, but good enough for demo)
        lat_diff = abs(depot1.coordinates.get("lat", 0) - depot2.coordinates.get("lat", 0))
        lon_diff = abs(depot1.coordinates.get("lon", 0) - depot2.coordinates.get("lon", 0))
        # Rough conversion: 1 degree ≈ 111 km
        distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111
        return max(distance_km, 5.0)  # Minimum 5 km
    return 20.0  # Default 20 km if coordinates not available

