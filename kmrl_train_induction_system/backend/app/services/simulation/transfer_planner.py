"""Inter-depot transfer planner for optimizing global fleet distribution."""
from typing import Dict, List
from app.models.depot import DepotConfig, DepotSimulationResult, TransferRecommendation


def plan_transfers(
    depots_results: List[DepotSimulationResult],
    depot_configs: Dict[str, DepotConfig],
    global_required_service: int,
    transfer_budget_hours: float = 2.0,
) -> List[TransferRecommendation]:
    recommendations: List[TransferRecommendation] = []
    total_service = sum(result.stabling_summary.get("service_trains", 0) for result in depots_results)
    global_shortfall = max(0, global_required_service - total_service)
    if global_shortfall == 0:
        return recommendations

    excess_depots = []
    shortfall_depots = []
    for result in depots_results:
        config = depot_configs.get(result.depot_id)
        if not config:
            continue
        service_shortfall = int(result.stabling_summary.get("service_shortfall", 0) or 0)
        service_used = int(result.stabling_summary.get("service_trains", 0) or 0)
        excess_trains = max(0, len(result.assigned_trains) - service_used)
        destination_capacity = max(0, int(config.service_bays) - service_used)
        if excess_trains > 0:
            excess_depots.append({"depot_id": result.depot_id, "excess_trains": excess_trains, "result": result})
        if service_shortfall > 0 and destination_capacity > 0:
            shortfall_depots.append({
                "depot_id": result.depot_id,
                "shortfall": service_shortfall,
                "available_capacity": destination_capacity,
                "result": result,
            })

    for shortfall_info in shortfall_depots:
        target_id = shortfall_info["depot_id"]
        target_config = depot_configs[target_id]
        for source_info in excess_depots:
            source_id = source_info["depot_id"]
            if source_id == target_id:
                continue
            source_config = depot_configs[source_id]
            source_result = source_info["result"]
            service_ids = set(source_result.bay_layout_after.get("service", {}).values())
            transferable = [train_id for train_id in source_result.assigned_trains if train_id not in service_ids]
            count = min(shortfall_info["shortfall"], shortfall_info["available_capacity"], source_info["excess_trains"], len(transferable))
            for train_id in transferable[:count]:
                cost, benefit, dead_km, time_hours = _compute_transfer_metrics(source_config, target_config, train_id)
                if time_hours > transfer_budget_hours:
                    continue
                recommendations.append(TransferRecommendation(
                    from_depot=source_id,
                    to_depot=target_id,
                    train_id=train_id,
                    cost_estimate=cost,
                    benefit_estimate=benefit,
                    reason=f"Reduce service shortfall at {target_config.name}",
                    feasibility=True,
                    recommended=benefit > cost,
                    dead_km=dead_km,
                    estimated_time_hours=time_hours,
                ))
    recommendations.sort(key=lambda x: x.benefit_estimate / max(x.cost_estimate, 1), reverse=True)
    return recommendations


def _compute_transfer_metrics(from_depot: DepotConfig, to_depot: DepotConfig, train_id: str) -> tuple[float, float, float, float]:
    dead_km = _estimate_distance(from_depot, to_depot)
    cost = dead_km * 3000 + 2000 + dead_km * 50 + 10000
    benefit = 50000 * 0.5
    time_hours = (30 + dead_km * 2) / 60
    return cost, benefit, dead_km, time_hours


def _estimate_distance(depot1: DepotConfig, depot2: DepotConfig) -> float:
    if depot1.coordinates and depot2.coordinates:
        lat_diff = abs(depot1.coordinates.get("lat", 0) - depot2.coordinates.get("lat", 0))
        lon_diff = abs(depot1.coordinates.get("lon", 0) - depot2.coordinates.get("lon", 0))
        return max(((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111, 5.0)
    return 20.0
