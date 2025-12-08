"""Tests for Muttom depot stabling + shunting logic."""
import asyncio
import pytest

from app.services.stabling_optimizer import StablingGeometryOptimizer


@pytest.mark.asyncio
async def test_capacity_enforced_with_overflow():
    optimizer = StablingGeometryOptimizer()
    trainsets = []
    decisions = []

    # 25 trainsets: 13 service, 6 maintenance, 6 standby
    for idx in range(25):
        train_id = f"T-{idx:03d}"
        trainsets.append(
            {
                "trainset_id": train_id,
                "current_location": {"depot": "Muttom Depot", "bay": (idx % 12) + 1},
                "job_cards": {"critical_cards": 0},
                "current_mileage": 10000 + idx * 100,
            }
        )
        if idx < 13:
            decisions.append({"trainset_id": train_id, "decision": "SERVICE", "score": 1.0 - idx * 0.01})
        elif idx < 19:
            severity = "HEAVY" if idx % 2 == 0 else "LIGHT"
            decisions.append({"trainset_id": train_id, "decision": "MAINTENANCE", "maintenance_severity": severity})
        else:
            decisions.append({"trainset_id": train_id, "decision": "STANDBY"})

    result = await optimizer.optimize_stabling_geometry(trainsets, decisions)
    summary = result["stabling_summary"]

    assert summary["stabled_service_trains"] == 6
    assert summary["stabled_maintenance_trains"] == 4
    assert summary["stabled_standby_trains"] == 2
    assert summary["total_stabled_trains"] <= 12
    assert summary["unassigned_due_to_capacity"] > 0


@pytest.mark.asyncio
async def test_service_shortfall_is_capacity_based():
    optimizer = StablingGeometryOptimizer()
    trainsets = [{"trainset_id": f"T-{i:03d}", "current_location": {"depot": "Muttom Depot", "bay": i + 1}} for i in range(15)]
    decisions = [{"trainset_id": t["trainset_id"], "decision": "SERVICE", "score": 1.0} for t in trainsets]

    result = await optimizer.optimize_stabling_geometry(trainsets, decisions, fleet_req={"required_service_trains": 13})
    requirement = result["service_requirement"]

    assert requirement["required_service_trains"] == 13
    assert requirement["stabled_service_trains"] >= 13  # depot + terminals
    assert requirement["capacity_shortfall"] == 0
    assert requirement["effective_service_shortfall"] == 0


@pytest.mark.asyncio
async def test_bay_diff_categorizes_move_enter_exit():
    optimizer = StablingGeometryOptimizer()
    trainsets = [
        {"trainset_id": "T-MOVE", "current_location": {"depot": "Muttom Depot", "bay": 5}},
        {"trainset_id": "T-ENTER", "current_location": {"depot": "Muttom Depot", "bay": None}},
        {"trainset_id": "T-EXIT", "current_location": {"depot": "Muttom Depot", "bay": 2}},
        {"trainset_id": "T-SB-HI1", "current_location": {"depot": "Muttom Depot", "bay": None}},
        {"trainset_id": "T-SB-HI2", "current_location": {"depot": "Muttom Depot", "bay": None}},
    ]
    decisions = [
        {"trainset_id": "T-MOVE", "decision": "SERVICE", "score": 0.9},
        {"trainset_id": "T-ENTER", "decision": "SERVICE", "score": 0.8},
        {"trainset_id": "T-EXIT", "decision": "STANDBY", "priority": 0},
        {"trainset_id": "T-SB-HI1", "decision": "STANDBY", "priority": 10},
        {"trainset_id": "T-SB-HI2", "decision": "STANDBY", "priority": 9},
    ]

    result = await optimizer.optimize_stabling_geometry(trainsets, decisions)
    diff = result["bay_diff"]
    move_types = {d["trainset_id"]: d["move_type"] for d in diff}

    assert move_types["T-MOVE"] == "MOVE"
    assert move_types["T-ENTER"] == "ENTER"
    assert move_types["T-EXIT"] == "EXIT"


@pytest.mark.asyncio
async def test_shunting_feasibility_flips_when_window_exceeded():
    optimizer = StablingGeometryOptimizer()
    # Stretch bay positions to force large shunting times
    optimizer.depot_layouts["Muttom Depot"]["bay_positions"] = {
        1: {"x": 0, "y": 0},
        2: {"x": 10, "y": 0},
        7: {"x": 1000, "y": 0},
        8: {"x": 1000, "y": 1000},
    }

    trainsets = [
        {"trainset_id": "T-001", "current_location": {"depot": "Muttom Depot", "bay": 1}},
        {"trainset_id": "T-002", "current_location": {"depot": "Muttom Depot", "bay": 2}},
    ]
    decisions = [
        {"trainset_id": "T-001", "decision": "SERVICE", "score": 1.0},
        {"trainset_id": "T-002", "decision": "SERVICE", "score": 0.9},
    ]

    result = await optimizer.optimize_stabling_geometry(trainsets, decisions)
    shunting_summary = result["shunting_summary"]

    assert shunting_summary["feasible"] is False
    assert shunting_summary["total_time_min"] > optimizer.operational_window["minutes"]