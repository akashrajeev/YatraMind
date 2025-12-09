"""
Tests for inter-depot transfer planner
"""
import pytest
from app.models.depot import DepotConfig, DepotSimulationResult, LocationType
from app.services.simulation.transfer_planner import plan_transfers


def test_transfer_recommendation():
    """Test that beneficial transfers are recommended"""
    # Create depot configs
    depot1 = DepotConfig(
        depot_id="DEPOT1",
        name="Depot 1",
        location_type=LocationType.FULL_DEPOT,
        service_bays=6,
        maintenance_bays=4,
        standby_bays=2
    )
    
    depot2 = DepotConfig(
        depot_id="DEPOT2",
        name="Depot 2",
        location_type=LocationType.FULL_DEPOT,
        service_bays=6,
        maintenance_bays=3,
        standby_bays=2
    )
    
    # Create results: depot1 has excess, depot2 has shortfall
    result1 = DepotSimulationResult(
        depot_id="DEPOT1",
        depot_name="Depot 1",
        assigned_trains=["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
        stabling_summary={
            "service_trains": 6,
            "maintenance_trains": 2,
            "standby_trains": 0,
            "service_shortfall": 0,
            "capacity_shortfall": 0
        },
        bay_layout_before={},
        bay_layout_after={"service": {"BAY_1": "T1", "BAY_2": "T2"}},
        bay_diff=[],
        shunting_operations=[],
        shunting_summary={"total_time_min": 0, "feasible": True},
        kpis={}
    )
    
    result2 = DepotSimulationResult(
        depot_id="DEPOT2",
        depot_name="Depot 2",
        assigned_trains=["T9", "T10"],
        stabling_summary={
            "service_trains": 2,
            "maintenance_trains": 0,
            "standby_trains": 0,
            "service_shortfall": 4,  # Needs 4 more
            "capacity_shortfall": 0
        },
        bay_layout_before={},
        bay_layout_after={"service": {"BAY_1": "T9", "BAY_2": "T10"}},
        bay_diff=[],
        shunting_operations=[],
        shunting_summary={"total_time_min": 0, "feasible": True},
        kpis={}
    )
    
    recommendations = plan_transfers(
        depots_results=[result1, result2],
        depot_configs={"DEPOT1": depot1, "DEPOT2": depot2},
        global_required_service=10
    )
    
    # Should recommend transfers from depot1 to depot2
    assert len(recommendations) > 0
    assert any(r.from_depot == "DEPOT1" and r.to_depot == "DEPOT2" for r in recommendations)

