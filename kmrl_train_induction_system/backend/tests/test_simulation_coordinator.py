"""
Tests for simulation coordinator
"""
import pytest
from app.models.depot import DepotConfig, LocationType
from app.services.simulation.coordinator import run_simulation


def test_deterministic_simulation():
    """Test that simulations with same seed produce same results"""
    depots = [
        DepotConfig(
            depot_id="MUTTOM",
            name="Muttom Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bays=6,
            maintenance_bays=4,
            standby_bays=2
        )
    ]
    
    result1 = run_simulation(
        depots=depots,
        fleet_count=25,
        seed=12345,
        ai_mode=False
    )
    
    result2 = run_simulation(
        depots=depots,
        fleet_count=25,
        seed=12345,
        ai_mode=False
    )
    
    # Should produce same run_id and same assignments
    assert result1.run_id == result2.run_id
    assert result1.per_depot["MUTTOM"].assigned_trains == result2.per_depot["MUTTOM"].assigned_trains


def test_multi_depot_simulation():
    """Test simulation with multiple depots"""
    depots = [
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
    
    result = run_simulation(
        depots=depots,
        fleet_count=40,
        seed=12345,
        ai_mode=False
    )
    
    assert len(result.per_depot) == 2
    assert "MUTTOM" in result.per_depot
    assert "KAKKANAD" in result.per_depot
    
    # Check that trains are distributed
    total_assigned = sum(
        len(res.assigned_trains) for res in result.per_depot.values()
    )
    assert total_assigned == 40


def test_capacity_validation():
    """Test that capacity shortfall is detected"""
    depots = [
        DepotConfig(
            depot_id="SMALL",
            name="Small Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bays=2,
            maintenance_bays=1,
            standby_bays=1
        )
    ]
    
    result = run_simulation(
        depots=depots,
        fleet_count=100,  # Way more than capacity
        seed=12345,
        ai_mode=False
    )
    
    # Should have warnings about capacity
    assert len(result.warnings) > 0
    assert any("capacity" in w.lower() for w in result.warnings)


def test_service_requirement_auto_compute():
    """Test automatic service requirement computation"""
    depots = [
        DepotConfig(
            depot_id="MUTTOM",
            name="Muttom Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bays=6,
            maintenance_bays=4,
            standby_bays=2
        )
    ]
    
    result = run_simulation(
        depots=depots,
        fleet_count=40,
        service_requirement=None,  # Auto-compute
        seed=12345,
        ai_mode=False
    )
    
    # Should have computed service requirement
    assert result.config_snapshot["service_requirement"] is not None
    assert result.config_snapshot["service_requirement"] > 0

