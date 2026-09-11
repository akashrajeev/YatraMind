"""Application service for assignment reads and state transitions."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.models.assignment import Assignment, AssignmentSummary, AssignmentStatus
from app.models.trainset import InductionDecision
from app.repositories.mongo_assignments import MongoAssignmentRepository


class AssignmentService:
    """Coordinate assignment persistence and API-facing domain models."""

    def __init__(self, repository: MongoAssignmentRepository | None = None) -> None:
        self.repository = repository or MongoAssignmentRepository()

    @staticmethod
    def to_model(document: Mapping[str, Any]) -> Assignment:
        doc = dict(document)
        trainset_id = str(doc.get("trainset_id", ""))
        decision = dict(doc.get("decision") or {})
        decision.setdefault("trainset_id", trainset_id)
        if "reasoning" in decision and "reasons" not in decision:
            reasoning = decision.pop("reasoning")
            decision["reasons"] = [reasoning] if reasoning else []
        decision.setdefault("decision", "STANDBY")
        decision.setdefault("confidence_score", 0.8)
        decision.setdefault("reasons", [])
        decision.setdefault("score", 0.0)
        decision.setdefault("top_reasons", [])
        decision.setdefault("top_risks", [])
        decision.setdefault("violations", [])
        decision.setdefault("shap_values", [])
        doc["decision"] = decision
        doc.setdefault("created_by", doc.get("assigned_to", "system"))

        for source, target in (("updated_at", "last_updated"), ("scheduled_date", "execution_date")):
            if source in doc and target not in doc:
                value = doc[source]
                if isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        value = None
                if value is not None:
                    doc[target] = value
        if isinstance(doc.get("created_at"), str):
            try:
                doc["created_at"] = datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00"))
            except ValueError:
                doc["created_at"] = datetime.now(timezone.utc)
        return Assignment(**doc)

    async def list_assignments(
        self,
        *,
        status: AssignmentStatus | None = None,
        trainset_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        priority: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Assignment]:
        documents = await self.repository.list(
            status=status.value if status else None,
            trainset_id=trainset_id,
            created_after=created_after,
            created_before=created_before,
            priority=priority,
            limit=limit,
            offset=offset,
        )
        result: list[Assignment] = []
        for document in documents:
            try:
                result.append(self.to_model(document))
            except Exception:
                continue
        return result

    async def get(self, assignment_id: str) -> Assignment | None:
        document = await self.repository.get(assignment_id)
        return self.to_model(document) if document else None

    async def summary(self) -> AssignmentSummary:
        return AssignmentSummary(
            total_assignments=await self.repository.total_count(),
            pending_count=await self.repository.count_status(AssignmentStatus.PENDING.value),
            approved_count=await self.repository.count_status(AssignmentStatus.APPROVED.value),
            rejected_count=await self.repository.count_status(AssignmentStatus.REJECTED.value),
            overridden_count=await self.repository.count_status(AssignmentStatus.OVERRIDDEN.value),
            high_priority_count=await self.repository.count_high_priority(),
            critical_risks_count=await self.repository.count_with_violations(),
            avg_confidence_score=await self.repository.average_confidence(),
            last_updated=datetime.now(timezone.utc),
        )

    async def create(self, assignment: Assignment) -> Assignment:
        await self.repository.insert(assignment.dict())
        return assignment

    async def approve(
        self,
        assignment_ids: Sequence[str],
        user_id: str,
        comments: str | None,
    ) -> int:
        return await self.repository.approve_pending(
            assignment_ids=assignment_ids,
            user_id=user_id,
            comments=comments,
            when=datetime.now(timezone.utc),
        )

    async def override(
        self,
        assignment_id: str,
        user_id: str,
        reason: str,
        override_decision: str,
    ) -> Mapping[str, Any] | None:
        return await self.repository.override_pending(
            assignment_id=assignment_id,
            user_id=user_id,
            reason=reason,
            override_decision=override_decision,
            when=datetime.now(timezone.utc),
        )
