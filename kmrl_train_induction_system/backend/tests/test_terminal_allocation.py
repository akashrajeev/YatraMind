import asyncio
import pytest

from app.services.stabling_optimizer import StablingGeometryOptimizer


def _build_train(train_id, decision, first_departure="Aluva", bay=None):
    return (
        {"trainset_id": train_id, "current_location": {"depot": "Muttom Depot", "bay": bay}},
        {"trainset_id": train_id, "decision": decision, "score": 1.0, "first_departure_station": first_departure},
    )


@pytest.mark.asyncio
async def test_terminal_allocation_with_overflow_distribution():
    optimizer = StablingGeometryOptimizer()
    trainsets = []
    decisions = []

    # 13 service, 6 standby, 6 maintenance
    for i in range(13):
        t, d = _build_train(f"SVC-{i:02d}", "SERVICE", "Aluva" if i % 2 == 0 else "Petta", (i % 12) + 1)
        trainsets.append(t)
        decisions.append(d)
    for i in range(6):
        t, d = _build_train(f"STB-{i:02d}", "STANDBY", "Aluva", None)
        trainsets.append(t)
        decisions.append(d)
    for i in range(6):
        t, d = _build_train(f"MTN-{i:02d}", "MAINTENANCE", "Aluva", None)
        trainsets.append(t)
        decisions.append(d)

    result = await optimizer.optimize_stabling_geometry(trainsets, decisions, {"required_service_trains": 13})

    stabling = result["stabling_summary"]
    terminals = result["terminal_allocation"]
    overflow = result["overflow_summary"]

    # Muttom bay caps respected
    assert stabling["stabled_service_trains"] == 6
    assert stabling["stabled_maintenance_trains"] == 4
    assert stabling["stabled_standby_trains"] == 2

    # Terminals pick up remaining service
    service_used = terminals["Aluva Terminal"]["service_used"] + terminals["Petta Terminal"]["service_used"]
    assert service_used >= 7  # overflow service rakes sent to terminals

    # Overflow counts reported
    assert overflow["unassigned_after_muttom"] >= 13  # anything beyond 12 bays
    assert overflow["unassigned_after_terminals"] >= 0


@pytest.mark.asyncio
async def test_terminal_capacity_enough_no_unassigned_after_terminals():
    optimizer = StablingGeometryOptimizer()
    trainsets = []
    decisions = []

    for i in range(10):
        t, d = _build_train(f"SVC-{i:02d}", "SERVICE", "Aluva", None)
        trainsets.append(t)
        decisions.append(d)
    for i in range(4):
        t, d = _build_train(f"STB-{i:02d}", "STANDBY", "Petta", None)
        trainsets.append(t)
        decisions.append(d)

    res = await optimizer.optimize_stabling_geometry(trainsets, decisions, {"required_service_trains": 10})
    assert res["overflow_summary"]["unassigned_after_terminals"] == 0


@pytest.mark.asyncio
async def test_terminal_capacity_small_creates_unassigned():
    optimizer = StablingGeometryOptimizer()
    # Force tiny terminal capacity
    optimizer.terminal_layouts["Aluva Terminal"]["service_stabling_capacity"] = 1
    optimizer.terminal_layouts["Petta Terminal"]["service_stabling_capacity"] = 0

    trainsets = []
    decisions = []
    for i in range(12):
        t, d = _build_train(f"SVC-{i:02d}", "SERVICE", "Aluva", None)
        trainsets.append(t)
        decisions.append(d)

    res = await optimizer.optimize_stabling_geometry(trainsets, decisions, {"required_service_trains": 12})
    assert res["overflow_summary"]["unassigned_after_terminals"] > 0


@pytest.mark.asyncio
async def test_rollout_plan_has_start_locations():
    optimizer = StablingGeometryOptimizer()
    trainsets = []
    decisions = []
    for i in range(13):
        t, d = _build_train(f"SVC-{i:02d}", "SERVICE", "Petta" if i % 2 else "Aluva", (i % 12) + 1)
        trainsets.append(t)
        decisions.append(d)

    res = await optimizer.optimize_stabling_geometry(trainsets, decisions, {"required_service_trains": 13})
    rollout = res["service_rollout_plan"]

    assert len([r for r in rollout if r["start_location"] == "Muttom Depot"]) == 6
    assert len(rollout) >= 13  # includes terminals
    assert all(r["start_location"] in ["Muttom Depot", "Aluva Terminal", "Petta Terminal"] for r in rollout)

