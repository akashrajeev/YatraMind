from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import logging
import uuid
import random
import functools
import traceback

from app.models.assignment import (
    Assignment,
    AssignmentCreate,
    AssignmentSummary,
    ApprovalRequest,
    OverrideRequest,
    AssignmentStatus,
)
from app.models.audit import AuditLogCreate, AuditAction
from app.models.trainset import InductionDecision
from app.security import require_api_key
from app.services.auth_service import require_role, get_current_user
from app.models.user import UserRole, User
from app.services.notification_service import NotificationService
from app.utils.explainability import generate_maintenance_reasons
from app.utils.cloud_database import cloud_db_manager
from app.services.assignment_service import AssignmentService

logger = logging.getLogger(__name__)
router = APIRouter()
assignment_service = AssignmentService()
notification_service = NotificationService()


def safe_background_task(func):
    """Safely execute an async background task and log failures."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            logger.error("Error in background task %s: %s", func.__name__, exc)
            logger.error(traceback.format_exc())
    return wrapper


# Kept for compatibility with existing tests/importers.
def _transform_doc_to_assignment(doc: Dict[str, Any]) -> Assignment:
    return assignment_service.to_model(doc)


@router.get("/", response_model=List[Assignment])
async def get_assignments(
    status: Optional[AssignmentStatus] = None,
    trainset_id: Optional[str] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    priority: Optional[int] = None,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
):
    """Get assignments with filtering and pagination."""
    try:
        return await assignment_service.list_assignments(
            status=status,
            trainset_id=trainset_id,
            created_after=created_after,
            created_before=created_before,
            priority=priority,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("Error fetching assignments: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


@router.get("/summary", response_model=AssignmentSummary)
async def get_assignment_summary(_auth=Depends(require_api_key)):
    """Get assignment summary statistics from the repository."""
    try:
        return await assignment_service.summary()
    except Exception as exc:
        logger.error("Error fetching assignment summary: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


def _needs_maintenance(trainset: Dict[str, Any]) -> bool:
    """Check whether a trainset is blocked by maintenance conditions."""
    job_cards = trainset.get("job_cards", {})
    if not isinstance(job_cards, dict):
        job_cards = {}
    try:
        critical_cards = int(job_cards.get("critical_cards", 0) or 0)
    except (TypeError, ValueError):
        critical_cards = 0
    try:
        current_mileage = float(trainset.get("current_mileage", 0.0) or 0.0)
    except (TypeError, ValueError):
        current_mileage = 0.0
    try:
        max_mileage = float(trainset.get("max_mileage_before_maintenance", float("inf")))
    except (TypeError, ValueError):
        max_mileage = float("inf")
    return critical_cards > 0 or (
        max_mileage > 0 and current_mileage >= max_mileage * 0.95
    )


@router.get("/conflicts", response_model=List[Assignment])
async def get_conflict_assignments(_auth=Depends(require_api_key)):
    """Get trains that need maintenance with maintenance reasons."""
    try:
        trainsets_collection = await cloud_db_manager.get_collection("trainsets")
        assignments = await cloud_db_manager.get_collection("assignments")
        maintenance_trainsets: list[Assignment] = []

        async for trainset_doc in trainsets_collection.find({}):
            try:
                trainset_doc = dict(trainset_doc)
                trainset_doc.pop("_id", None)
                trainset_id = trainset_doc.get("trainset_id", "")
                if not trainset_id or not _needs_maintenance(trainset_doc):
                    continue

                maintenance_info = generate_maintenance_reasons(trainset_doc)
                existing_assignment = await assignments.find_one({"trainset_id": trainset_id})
                if existing_assignment:
                    assignment = _transform_doc_to_assignment(existing_assignment)
                    assignment.decision.decision = "MAINTENANCE"
                    assignment.decision.top_reasons = maintenance_info.get("top_reasons", [])
                    assignment.decision.top_risks = maintenance_info.get("top_risks", [])
                    assignment.decision.score = 0.0
                    assignment.decision.confidence_score = 1.0
                else:
                    assignment = Assignment(
                        id=str(uuid.uuid4()),
                        trainset_id=trainset_id,
                        decision=InductionDecision(
                            trainset_id=trainset_id,
                            decision="MAINTENANCE",
                            confidence_score=1.0,
                            reasons=maintenance_info.get("top_reasons", []),
                            top_reasons=maintenance_info.get("top_reasons", []),
                            top_risks=maintenance_info.get("top_risks", []),
                            score=0.0,
                            violations=[],
                            shap_values=[],
                        ),
                        status=AssignmentStatus.PENDING,
                        created_by="system",
                        priority=5,
                    )
                maintenance_trainsets.append(assignment)
            except Exception as exc:
                logger.warning(
                    "Skipping trainset %s: %s",
                    trainset_doc.get("trainset_id", "unknown"),
                    exc,
                )

        maintenance_trainsets.sort(key=lambda item: item.priority, reverse=True)
        return maintenance_trainsets
    except Exception as exc:
        logger.error("Error fetching conflict assignments: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to fetch conflicts: {exc}")


@router.get("/{assignment_id}", response_model=Assignment)
async def get_assignment(assignment_id: str, _auth=Depends(require_api_key)):
    """Get a specific assignment by ID."""
    try:
        assignment = await assignment_service.get(assignment_id)
        if assignment is None:
            raise HTTPException(status_code=404, detail="Assignment not found")
        return assignment
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error fetching assignment %s: %s", assignment_id, exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


@router.post("/", response_model=Assignment)
async def create_assignment(
    assignment_data: AssignmentCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Create a new assignment."""
    try:
        assignment = Assignment(
            id=str(uuid.uuid4()),
            trainset_id=assignment_data.trainset_id,
            decision=assignment_data.decision,
            created_by=getattr(assignment_data, "created_by", None) or "system",
            priority=assignment_data.priority,
            execution_date=assignment_data.execution_date,
        )
        await assignment_service.create(assignment)

        audit_log = AuditLogCreate(
            user_id=assignment.created_by,
            action=AuditAction.ASSIGNMENT_CREATED,
            resource_type="assignment",
            resource_id=assignment.id,
            details={
                "trainset_id": assignment.trainset_id,
                "decision": assignment.decision.decision,
                "priority": assignment.priority,
            },
        )
        background_tasks.add_task(log_audit_event, audit_log)
        background_tasks.add_task(
            send_assignment_notification,
            assignment.id,
            "Assignment Created",
            f"New assignment created for trainset {assignment.trainset_id}",
        )
        return assignment
    except Exception as exc:
        logger.error("Error creating assignment: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to create assignment: {exc}")


@router.post("/approve")
async def approve_assignments(
    approval_request: ApprovalRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Approve multiple pending assignments."""
    try:
        user_id = approval_request.user_id or "system"
        modified_count = await assignment_service.approve(
            approval_request.assignment_ids,
            user_id,
            approval_request.comments,
        )
        if modified_count == 0:
            raise HTTPException(status_code=404, detail="No pending assignments found to approve")

        audit_log = AuditLogCreate(
            user_id=user_id,
            action=AuditAction.ASSIGNMENT_APPROVED,
            resource_type="assignment",
            resource_id=",".join(approval_request.assignment_ids),
            details={"assignment_count": modified_count, "comments": approval_request.comments},
        )
        background_tasks.add_task(log_audit_event, audit_log)
        for assignment_id in approval_request.assignment_ids:
            background_tasks.add_task(
                send_assignment_notification,
                assignment_id,
                "Assignment Approved",
                "Assignment has been approved and locked for execution",
            )
        return {
            "message": f"Successfully approved {modified_count} assignments",
            "approved_count": modified_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error approving assignments: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to approve assignments: {exc}")


@router.post("/override")
async def override_assignment(
    override_request: OverrideRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Override a pending assignment decision."""
    try:
        user_id = override_request.user_id or "system"
        updated = await assignment_service.override(
            assignment_id=override_request.assignment_id,
            user_id=user_id,
            reason=override_request.reason,
            override_decision=override_request.override_decision,
        )
        if updated is None:
            existing = await assignment_service.get(override_request.assignment_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="Assignment not found")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot override assignment with status {existing.status}",
            )

        audit_log = AuditLogCreate(
            user_id=user_id,
            action=AuditAction.ASSIGNMENT_OVERRIDDEN,
            resource_type="assignment",
            resource_id=override_request.assignment_id,
            details={
                "original_decision": updated.get("decision", {}).get("decision", "UNKNOWN"),
                "override_decision": override_request.override_decision,
                "reason": override_request.reason,
                "trainset_id": updated.get("trainset_id"),
            },
            risk_level="HIGH",
        )
        background_tasks.add_task(log_audit_event, audit_log)
        background_tasks.add_task(
            send_assignment_notification,
            override_request.assignment_id,
            "Assignment Overridden",
            f"Assignment decision overridden: {override_request.override_decision}",
        )
        return {
            "message": "Assignment successfully overridden",
            "assignment_id": override_request.assignment_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error overriding assignment: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to override assignment: {exc}")


# Legacy seed helper retained for local/demo environments. It is intentionally
# outside the normal request persistence path and will be moved to scripts.
async def create_sample_assignments():
    """Create sample assignments with real data when explicitly invoked."""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        if await collection.count_documents({}) > 0:
            return
        trainsets_collection = await cloud_db_manager.get_collection("trainsets")
        trainsets = [doc async for doc in trainsets_collection.find({}).limit(15)]
        if not trainsets:
            sample_trainsets = [
                {"trainset_id": f"TS-{i:03d}", "status": "ACTIVE", "sensor_health_score": 0.85 + (i * 0.01)}
                for i in range(1, 16)
            ]
            await trainsets_collection.insert_many(sample_trainsets)
            trainsets = sample_trainsets
        assignments = []
        for i, trainset in enumerate(trainsets[:15]):
            trainset_id = trainset.get("trainset_id", f"TS-{i:03d}")
            decision = random.choice(["INDUCT", "STANDBY", "MAINTENANCE"])
            violations = []
            if random.random() < 0.4:
                violation_types = [
                    "Safety certificate expiring soon",
                    "Maintenance overdue",
                    "Cleaning schedule conflict",
                    "Branding contract expired",
                    "High failure risk detected",
                    "Sensor health below threshold",
                    "Certificate validation failed",
                ]
                violations = random.sample(violation_types, random.randint(1, 3))
            assignments.append({
                "id": f"ASS-{i+1:03d}",
                "trainset_id": trainset_id,
                "status": "PENDING",
                "priority": random.randint(1, 5),
                "decision": {
                    "trainset_id": trainset_id,
                    "decision": decision,
                    "confidence_score": round(random.uniform(0.7, 0.95), 2),
                    "reasons": [f"AI decision based on trainset {trainset_id} analysis"],
                    "top_reasons": [],
                    "top_risks": [],
                    "violations": violations,
                    "score": 0.0,
                    "shap_values": [],
                },
                "created_at": datetime.now(timezone.utc) - timedelta(days=random.randint(0, 7)),
                "last_updated": datetime.now(timezone.utc),
                "created_by": f"system-{random.randint(1, 5)}",
                "execution_date": datetime.now(timezone.utc) + timedelta(days=random.randint(1, 3)),
            })
        await collection.insert_many(assignments)
    except Exception as exc:
        logger.error("Error creating sample assignments: %s", exc)


@safe_background_task
async def log_audit_event(audit_log: AuditLogCreate):
    """Log an audit event."""
    try:
        audit_collection = await cloud_db_manager.get_collection("audit_logs")
        audit_doc = audit_log.dict()
        audit_doc["id"] = str(uuid.uuid4())
        audit_doc["timestamp"] = datetime.now(timezone.utc)
        await audit_collection.insert_one(audit_doc)
    except Exception as exc:
        logger.error("Failed to log audit event: %s", exc)


@safe_background_task
async def send_assignment_notification(assignment_id: str, title: str, message: str):
    """Send an assignment notification."""
    try:
        await notification_service.create_notification(
            type="ASSIGNMENT_UPDATE",
            priority="MEDIUM",
            title=title,
            message=message,
            data={"assignment_id": assignment_id},
        )
    except Exception as exc:
        logger.error("Failed to send assignment notification: %s", exc)
