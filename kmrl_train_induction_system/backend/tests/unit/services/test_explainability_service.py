import pytest

from app.services.explainability_service import ExplainabilityService


@pytest.mark.asyncio
async def test_explainability_is_non_blocking_by_default():
    result = await ExplainabilityService().explain(
        {"trainset_id": "T-001"},
        "INDUCT",
    )
    assert result["trainset_id"] == "T-001"
    assert result["decision"] == "INDUCT"
    assert result["explainability_available"] is False
