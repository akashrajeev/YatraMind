import pytest

from app.models.assignment import AssignmentStatus
from app.services.assignment_service import AssignmentService


class FakeAssignmentRepository:
    async def list(self, **kwargs):
        return [{
            "id": "A-1",
            "trainset_id": "T-001",
            "decision": {"decision": "INDUCT", "confidence_score": 0.91},
            "status": "PENDING",
            "priority": 4,
        }]

    async def get(self, assignment_id):
        return {
            "id": assignment_id,
            "trainset_id": "T-001",
            "decision": {"decision": "INDUCT", "confidence_score": 0.91},
            "status": "PENDING",
            "priority": 4,
        }

    async def insert(self, payload):
        return payload["id"]

    async def total_count(self):
        return 3

    async def count_status(self, status):
        return {"PENDING": 1, "APPROVED": 1, "REJECTED": 1, "OVERRIDDEN": 0}[status]

    async def count_high_priority(self, minimum=4):
        return 1

    async def average_confidence(self):
        return 0.91

    async def count_with_violations(self):
        return 0

    async def approve_pending(self, assignment_ids, user_id, comments, when):
        return len(assignment_ids)

    async def override_pending(self, assignment_id, user_id, reason, override_decision, when):
        return {
            "id": assignment_id,
            "trainset_id": "T-001",
            "decision": {"decision": "INDUCT", "confidence_score": 0.91},
            "status": "OVERRIDDEN",
        }

    async def list_assigned_trainset_ids(self):
        return set()

    async def get_by_trainset(self, trainset_id):
        return None


@pytest.mark.asyncio
async def test_assignment_service_list_uses_repository_and_models_results():
    service = AssignmentService(FakeAssignmentRepository())
    result = await service.list_assignments(status=AssignmentStatus.PENDING, limit=10)

    assert len(result) == 1
    assert result[0].id == "A-1"
    assert result[0].trainset_id == "T-001"


@pytest.mark.asyncio
async def test_assignment_service_summary_uses_repository_counts():
    service = AssignmentService(FakeAssignmentRepository())
    result = await service.summary()

    assert result.total_assignments == 3
    assert result.pending_count == 1
    assert result.approved_count == 1
    assert result.avg_confidence_score == 0.91
