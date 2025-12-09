"""
Infrastructure advisor for suggesting depot capacity expansions
"""
from typing import Dict, List

from app.models.depot import DepotConfig, DepotSimulationResult


class InfrastructureRecommendation:
    """Infrastructure expansion recommendation"""
    def __init__(
        self,
        depot_id: str,
        depot_name: str,
        bay_type: str,  # "service", "maintenance", "standby"
        bays_to_add: int,
        estimated_cost: float,
        shortfall_reduction: int,
        payback_days: float,
        roi: float
    ):
        self.depot_id = depot_id
        self.depot_name = depot_name
        self.bay_type = bay_type
        self.bays_to_add = bays_to_add
        self.estimated_cost = estimated_cost
        self.shortfall_reduction = shortfall_reduction
        self.payback_days = payback_days
        self.roi = roi


def suggest_infrastructure(
    depot_results: Dict[str, DepotSimulationResult],
    depot_configs: Dict[str, DepotConfig],
    cost_params: Dict[str, float]
) -> List[InfrastructureRecommendation]:
    """
    Suggest infrastructure expansions to eliminate shortfalls
    
    Args:
        depot_results: Results from depot simulations
        depot_configs: Depot configuration map
        cost_params: Cost parameters (from config)
    
    Returns:
        List of infrastructure recommendations sorted by ROI
    """
    recommendations = []
    
    for depot_id, result in depot_results.items():
        depot_config = depot_configs.get(depot_id)
        if not depot_config:
            continue
        
        stabling_summary = result.stabling_summary
        
        # Check service shortfall
        service_shortfall = stabling_summary.get("service_shortfall", 0)
        if service_shortfall > 0:
            rec = _compute_service_bay_recommendation(
                depot_config, service_shortfall, cost_params
            )
            if rec:
                recommendations.append(rec)
        
        # Check capacity shortfall
        capacity_shortfall = stabling_summary.get("capacity_shortfall", 0)
        if capacity_shortfall > 0:
            # Suggest standby bays for capacity
            rec = _compute_standby_bay_recommendation(
                depot_config, capacity_shortfall, cost_params
            )
            if rec:
                recommendations.append(rec)
        
        # Check maintenance shortfall (if maintenance trains exceed capacity)
        maintenance_shortfall = max(0, 
            stabling_summary.get("maintenance_trains", 0) - 
            depot_config.maintenance_bays
        )
        if maintenance_shortfall > 0:
            rec = _compute_maintenance_bay_recommendation(
                depot_config, maintenance_shortfall, cost_params
            )
            if rec:
                recommendations.append(rec)
    
    # Sort by ROI (descending)
    recommendations.sort(key=lambda x: x.roi, reverse=True)
    
    return recommendations


def _compute_service_bay_recommendation(
    depot: DepotConfig,
    shortfall: int,
    cost_params: Dict[str, float]
) -> InfrastructureRecommendation:
    """Compute service bay expansion recommendation"""
    bays_to_add = shortfall  # Add enough bays to cover shortfall
    cost_per_bay = cost_params.get("cost_per_service_bay", 5000000)
    total_cost = bays_to_add * cost_per_bay
    
    value_per_train_per_day = cost_params.get("value_per_train_per_day", 50000)
    daily_benefit = shortfall * value_per_train_per_day
    
    if daily_benefit > 0:
        payback_days = total_cost / daily_benefit
        roi = (daily_benefit * 365) / total_cost  # Annual ROI
    else:
        payback_days = float('inf')
        roi = 0.0
    
    return InfrastructureRecommendation(
        depot_id=depot.depot_id,
        depot_name=depot.name,
        bay_type="service",
        bays_to_add=bays_to_add,
        estimated_cost=total_cost,
        shortfall_reduction=shortfall,
        payback_days=payback_days,
        roi=roi
    )


def _compute_standby_bay_recommendation(
    depot: DepotConfig,
    shortfall: int,
    cost_params: Dict[str, float]
) -> InfrastructureRecommendation:
    """Compute standby bay expansion recommendation"""
    bays_to_add = shortfall
    cost_per_bay = cost_params.get("cost_per_standby_bay", 3000000)
    total_cost = bays_to_add * cost_per_bay
    
    # Standby bays have lower value (indirect benefit)
    value_per_train_per_day = cost_params.get("value_per_train_per_day", 50000) * 0.3
    daily_benefit = shortfall * value_per_train_per_day
    
    if daily_benefit > 0:
        payback_days = total_cost / daily_benefit
        roi = (daily_benefit * 365) / total_cost
    else:
        payback_days = float('inf')
        roi = 0.0
    
    return InfrastructureRecommendation(
        depot_id=depot.depot_id,
        depot_name=depot.name,
        bay_type="standby",
        bays_to_add=bays_to_add,
        estimated_cost=total_cost,
        shortfall_reduction=shortfall,
        payback_days=payback_days,
        roi=roi
    )


def _compute_maintenance_bay_recommendation(
    depot: DepotConfig,
    shortfall: int,
    cost_params: Dict[str, float]
) -> InfrastructureRecommendation:
    """Compute maintenance bay expansion recommendation"""
    bays_to_add = shortfall
    cost_per_bay = cost_params.get("cost_per_maintenance_bay", 8000000)
    total_cost = bays_to_add * cost_per_bay
    
    # Maintenance bays prevent downtime (high value)
    value_per_train_per_day = cost_params.get("value_per_train_per_day", 50000) * 0.8
    daily_benefit = shortfall * value_per_train_per_day
    
    if daily_benefit > 0:
        payback_days = total_cost / daily_benefit
        roi = (daily_benefit * 365) / total_cost
    else:
        payback_days = float('inf')
        roi = 0.0
    
    return InfrastructureRecommendation(
        depot_id=depot.depot_id,
        depot_name=depot.name,
        bay_type="maintenance",
        bays_to_add=bays_to_add,
        estimated_cost=total_cost,
        shortfall_reduction=shortfall,
        payback_days=payback_days,
        roi=roi
    )

