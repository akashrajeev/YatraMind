"""
Integration test: Compare 1-depot vs 2-depot simulation
"""
import pytest
from app.models.depot import DepotConfig, LocationType
from app.services.simulation.coordinator import run_simulation


def test_1_depot_vs_2_depots():
    """Test that 2-depot simulation reduces shortfall compared to 1-depot"""
    # Single depot simulation
    single_depot = [
        DepotConfig(
            depot_id="MUTTOM",
            name="Muttom Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bays=6,
            maintenance_bays=4,
            standby_bays=2
        )
    ]
    
    result_1depot = run_simulation(
        depots=single_depot,
        fleet_count=40,
        service_requirement=20,
        seed=12345,
        ai_mode=False
    )
    
    # Two depot simulation
    two_depots = [
        DepotConfig(
            depot_id="MUTTOM",
            name="Muttom Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bays=6,
            maintenance_bays=4,
            standby_bays=2
        ),
        DepotConfig(
            depot_id="KAKKANAD",
            name="Kakkanad Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bays=6,
            maintenance_bays=3,
            standby_bays=2
        )
    ]
    
    result_2depot = run_simulation(
        depots=two_depots,
        fleet_count=40,
        service_requirement=20,
        seed=12345,
        ai_mode=False
    )
    
    # Two depots should have lower or equal shortfall
    shortfall_1 = result_1depot.global_summary.get("effective_service_shortfall", 0)
    shortfall_2 = result_2depot.global_summary.get("effective_service_shortfall", 0)
    
    assert shortfall_2 <= shortfall_1, "Two depots should reduce or maintain shortfall"
    
    # Two depots should have more total service capacity
    total_service_1 = result_1depot.global_summary.get("total_service_trains", 0)
    total_service_2 = result_2depot.global_summary.get("total_service_trains", 0)
    
    assert total_service_2 >= total_service_1, "Two depots should provide equal or more service"

