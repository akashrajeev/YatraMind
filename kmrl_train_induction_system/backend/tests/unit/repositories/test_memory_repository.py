import pytest

from app.repositories.memory import MemoryOptimizationRepository, MemoryTrainsetRepository


@pytest.mark.asyncio
async def test_memory_trainset_repository_returns_copies():
    repo = MemoryTrainsetRepository([{"trainset_id": "T-001", "status": "STANDBY"}])
    item = await repo.get("T-001")
    item["status"] = "MAINTENANCE"
    stored = await repo.get("T-001")
    assert stored["status"] == "STANDBY"


@pytest.mark.asyncio
async def test_memory_optimization_repository_tracks_latest_run():
    repo = MemoryOptimizationRepository()
    await repo.save_run({"decisions": [{"trainset_id": "T-001", "decision": "INDUCT"}]})
    latest = await repo.get_latest_decisions()
    assert latest == [{"trainset_id": "T-001", "decision": "INDUCT"}]
