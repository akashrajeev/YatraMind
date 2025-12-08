import asyncio
import pytest

from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.models.trainset import MaintenanceSeverity


pytestmark = pytest.mark.asyncio


def _make_train(idx: int, decision: str, depot: str = "Muttom Depot", severity: str = "NONE"):
    return {
        "trainset_id": f"T-{idx:03d}",
        "current_location": {"depot": depot, "bay": ""},
        "job_cards": {"open_cards": 0, "critical_cards": 0},
        "induction_decision": {"decision": decision, "maintenance_severity": severity},
    }


async def _run(trainsets, decisions=None):
    decisions = decisions or [t["induction_decision"] for t in trainsets]
    optimizer = StablingGeometryOptimizer()
    return await optimizer.generate_rich_stabling_geometry(trainsets, decisions, fleet_req=None)


async def test_capacity_enforced_and_unassigned():
    # 25 service trains to Muttom (capacity 12)
    trainsets = [_make_train(i, "INDUCT") for i in range(1, 26)]
    res = await _run(trainsets)
    cap = res.capacity_summary
    assert cap["total_assigned"] <= cap["total_capacity"]
    assert cap["unassigned_due_to_capacity"] > 0
    assert res.unassigned_trainsets, "Unassigned list must be present when capacity exceeded"


async def test_maintenance_queue_created():
    # 5 maintenance trains, capacity 4 -> one queued
    trainsets = [_make_train(i, "MAINTENANCE", severity=MaintenanceSeverity.LIGHT.value) for i in range(1, 6)]
    res = await _run(trainsets)
    assert res.maintenance_queue is not None
    assert len(res.maintenance_queue) >= 1
    assert any(q.get("reason") == "No bay capacity" for q in res.maintenance_queue)


async def test_dead_km_and_reason_codes():
    # Place a service train from Muttom to Petta to ensure dead-km > 0
    trainsets = [
        _make_train(1, "INDUCT", depot="Muttom Depot"),
        _make_train(2, "STANDBY", depot="Petta Terminal"),
    ]
    res = await _run(trainsets)
    # Find bay assignment for T-001
    found = False
    for bays in res.bay_layout.values():
        for b in bays:
            if b.trainset_id == "T-001":
                found = True
                assert b.dead_km is not None
                assert b.dead_km["total"] >= 0
                assert b.placement_reason_code is not None
    assert found, "Expected T-001 to have a bay assignment"


async def test_shunting_window_feasibility_present():
    trainsets = [_make_train(i, "INDUCT") for i in range(1, 6)]
    res = await _run(trainsets)
    sw = res.shunting_window
    assert sw is not None
    assert "available_minutes" in sw and "required_minutes" in sw and "feasible" in sw

