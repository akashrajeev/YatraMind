"""Test snapshot capture functionality"""
import pytest
from app.utils.snapshot import capture_snapshot


@pytest.mark.asyncio
async def test_snapshot_contains_required_keys():
    """Test that snapshot contains all required keys"""
    snapshot = await capture_snapshot()
    
    required_keys = [
        "trainsets",
        "depot_layouts",
        "cleaning_slots",
        "certificates",
        "jobcards",
        "bay_locations",
        "config",
        "timestamp"
    ]
    
    for key in required_keys:
        assert key in snapshot, f"Missing required key: {key}"
    
    assert isinstance(snapshot["trainsets"], list)
    assert isinstance(snapshot["depot_layouts"], dict)
    assert isinstance(snapshot["cleaning_slots"], dict)
    assert isinstance(snapshot["certificates"], dict)
    assert isinstance(snapshot["jobcards"], dict)
    assert isinstance(snapshot["bay_locations"], dict)
    assert isinstance(snapshot["config"], dict)


@pytest.mark.asyncio
async def test_snapshot_is_deep_copy():
    """Test that snapshot is a deep copy and doesn't mutate original"""
    snapshot1 = await capture_snapshot()
    snapshot2 = await capture_snapshot()
    
    # Modify snapshot1
    if snapshot1["trainsets"]:
        snapshot1["trainsets"][0]["test_field"] = "modified"
    
    # snapshot2 should be unaffected
    if snapshot2["trainsets"]:
        assert "test_field" not in snapshot2["trainsets"][0]


@pytest.mark.asyncio
async def test_snapshot_trainsets_structure():
    """Test that trainsets in snapshot have expected structure"""
    snapshot = await capture_snapshot()
    
    for trainset in snapshot["trainsets"]:
        assert "trainset_id" in trainset
        assert isinstance(trainset["trainset_id"], str)







