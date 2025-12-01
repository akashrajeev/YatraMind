"""Unit tests for bay assignment conflicts and capacity limits"""
import pytest
from app.services.stabling_optimizer import StablingGeometryOptimizer


@pytest.fixture
def optimizer():
    return StablingGeometryOptimizer()


@pytest.fixture
def depot_layout():
    """Sample depot layout for testing"""
    return {
        "total_bays": 8,
        "maintenance_bays": [1, 2, 3],
        "cleaning_bays": [4, 5],
        "service_bays": [6, 7, 8],
        "bay_positions": {
            i: {"x": i * 10, "y": 0, "type": "service", "turnout_time": 5 + i}
            for i in range(1, 9)
        }
    }


def test_no_bay_conflicts_service_maintenance_standby(optimizer, depot_layout):
    """Test that service, maintenance, and standby trains don't conflict"""
    used_bays = set()
    
    # Assign 3 service trains (should get bays 6, 7, 8)
    service_trains = [{"trainset_id": f"S-{i}"} for i in range(1, 4)]
    service_assignments, _ = optimizer._assign_service_bays(service_trains, depot_layout, used_bays)
    used_bays.update(service_assignments.values())
    
    # Assign 2 maintenance trains (should get bays 1, 2)
    maintenance_trains = [{"trainset_id": f"M-{i}"} for i in range(1, 3)]
    maintenance_assignments, _ = optimizer._assign_maintenance_bays(maintenance_trains, depot_layout, used_bays)
    used_bays.update(maintenance_assignments.values())
    
    # Assign 3 standby trains (should get remaining bays 3, 4, 5)
    standby_trains = [{"trainset_id": f"ST-{i}"} for i in range(1, 4)]
    standby_assignments, _ = optimizer._assign_standby_bays(standby_trains, depot_layout, used_bays)
    used_bays.update(standby_assignments.values())
    
    # Check no duplicates
    all_assignments = {**service_assignments, **maintenance_assignments, **standby_assignments}
    all_bays = list(all_assignments.values())
    assert len(all_bays) == len(set(all_bays)), "Duplicate bay assignments found!"
    
    # Check service trains got service bays
    assert all(bay in depot_layout["service_bays"] for bay in service_assignments.values())
    
    # Check maintenance trains got maintenance bays
    assert all(bay in depot_layout["maintenance_bays"] for bay in maintenance_assignments.values())


def test_capacity_exceeded_service_bays(optimizer, depot_layout):
    """Test that overflow trains are marked as unassigned when capacity exceeded"""
    used_bays = set()
    
    # Try to assign 10 service trains to 3 service bays
    service_trains = [{"trainset_id": f"S-{i}"} for i in range(1, 11)]
    service_assignments, unassigned = optimizer._assign_service_bays(service_trains, depot_layout, used_bays)
    
    # Should only assign 3 trains (capacity)
    assert len(service_assignments) == 3
    
    # Should have 7 unassigned
    assert len(unassigned) == 7
    assert all(ts["reason"] == "no_capacity" for ts in unassigned)
    assert all("trainset_id" in ts for ts in unassigned)


def test_capacity_exceeded_maintenance_bays(optimizer, depot_layout):
    """Test maintenance bay capacity limits"""
    used_bays = set()
    
    # Try to assign 5 maintenance trains to 3 maintenance bays
    maintenance_trains = [{"trainset_id": f"M-{i}"} for i in range(1, 6)]
    maintenance_assignments, unassigned = optimizer._assign_maintenance_bays(maintenance_trains, depot_layout, used_bays)
    
    assert len(maintenance_assignments) == 3
    assert len(unassigned) == 2


def test_standby_uses_remaining_bays(optimizer, depot_layout):
    """Test that standby trains use remaining bays after service/maintenance"""
    used_bays = {6, 7, 8, 1, 2}  # Service and maintenance already assigned
    
    standby_trains = [{"trainset_id": f"ST-{i}"} for i in range(1, 4)]
    standby_assignments, unassigned = optimizer._assign_standby_bays(standby_trains, depot_layout, used_bays)
    
    # Should get remaining bays: 3, 4, 5
    assert len(standby_assignments) == 3
    assert set(standby_assignments.values()) == {3, 4, 5}
    assert len(unassigned) == 0


def test_no_duplicate_assignments_in_depot_optimization(optimizer):
    """Test that _optimize_depot_layout never creates duplicate bay assignments"""
    trainsets = [
        {"trainset_id": "T-001", "induction_decision": {"decision": "INDUCT"}},
        {"trainset_id": "T-002", "induction_decision": {"decision": "INDUCT"}},
        {"trainset_id": "T-003", "induction_decision": {"decision": "MAINTENANCE"}},
        {"trainset_id": "T-004", "induction_decision": {"decision": "MAINTENANCE"}},
        {"trainset_id": "T-005", "induction_decision": {"decision": "STANDBY"}},
        {"trainset_id": "T-006", "induction_decision": {"decision": "STANDBY"}},
    ]
    
    import asyncio
    result = asyncio.run(optimizer._optimize_depot_layout("Aluva", trainsets))
    
    bay_assignments = result["bay_assignments"]
    all_bays = list(bay_assignments.values())
    
    # Check no duplicates
    assert len(all_bays) == len(set(all_bays)), f"Duplicate bay assignments: {bay_assignments}"








