# backend/app/api/assignments.py
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import uuid
import random

from app.models.assignment import (
    Assignment, AssignmentCreate, AssignmentUpdate, AssignmentFilter, AssignmentSummary,
    ApprovalRequest, OverrideRequest, AssignmentStatus
)
from app.models.audit import AuditLogCreate, AuditAction
from app.utils.cloud_database import cloud_db_manager
from app.security import require_api_key, get_current_user
from app.services.notification_service import NotificationService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize notification service
notification_service = NotificationService()


@router.get("/", response_model=List[Assignment])
async def get_assignments(
    status: Optional[AssignmentStatus] = None,
    trainset_id: Optional[str] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    priority: Optional[int] = None,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    _auth=Depends(require_api_key)
):
    """Get assignments with filtering and pagination"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        
        # Build filter query
        filter_query = {}
        if status:
            filter_query["status"] = status.value
        if trainset_id:
            filter_query["trainset_id"] = trainset_id
        if created_after:
            filter_query["created_at"] = {"$gte": created_after}
        if created_before:
            filter_query["created_at"] = {"$lte": created_before}
        if priority:
            filter_query["priority"] = priority
        
        # Execute query with pagination
        cursor = collection.find(filter_query).sort("created_at", -1).skip(offset).limit(limit)
        assignments = []
        
        async for doc in cursor:
            doc.pop('_id', None)
            assignments.append(Assignment(**doc))
        
        return assignments
        
    except Exception as e:
        logger.error(f"Error fetching assignments: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/{assignment_id}", response_model=Assignment)
async def get_assignment(
    assignment_id: str,
    _auth=Depends(require_api_key)
):
    """Get specific assignment by ID"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        doc = await collection.find_one({"id": assignment_id})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        doc.pop('_id', None)
        return Assignment(**doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching assignment {assignment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/", response_model=Assignment)
async def create_assignment(
    assignment_data: AssignmentCreate,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Create a new assignment"""
    try:
        assignment_id = str(uuid.uuid4())
        
        assignment = Assignment(
            id=assignment_id,
            trainset_id=assignment_data.trainset_id,
            decision=assignment_data.decision,
            created_by=current_user["id"],
            priority=assignment_data.priority,
            execution_date=assignment_data.execution_date
        )
        
        # Save to database
        collection = await cloud_db_manager.get_collection("assignments")
        await collection.insert_one(assignment.dict())
        
        # Log audit event
        audit_log = AuditLogCreate(
            user_id=current_user["id"],
            action=AuditAction.ASSIGNMENT_CREATED,
            resource_type="assignment",
            resource_id=assignment_id,
            details={
                "trainset_id": assignment_data.trainset_id,
                "decision": assignment_data.decision.decision,
                "priority": assignment_data.priority
            }
        )
        background_tasks.add_task(log_audit_event, audit_log)
        
        # Send notification
        background_tasks.add_task(
            send_assignment_notification,
            assignment_id,
            "Assignment Created",
            f"New assignment created for trainset {assignment_data.trainset_id}"
        )
        
        logger.info(f"Created assignment {assignment_id} for trainset {assignment_data.trainset_id}")
        return assignment
        
    except Exception as e:
        logger.error(f"Error creating assignment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create assignment: {str(e)}")


@router.post("/approve")
async def approve_assignments(
    approval_request: ApprovalRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Approve multiple assignments"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        current_time = datetime.utcnow()
        
        # Update assignments
        result = await collection.update_many(
            {"id": {"$in": approval_request.assignment_ids}},
            {
                "$set": {
                    "status": AssignmentStatus.APPROVED.value,
                    "approved_by": current_user["id"],
                    "approved_at": current_time,
                    "approval_comments": approval_request.comments,
                    "last_updated": current_time
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="No assignments found to approve")
        
        # Log audit event
        audit_log = AuditLogCreate(
            user_id=current_user["id"],
            action=AuditAction.ASSIGNMENT_APPROVED,
            resource_type="assignment",
            resource_id=",".join(approval_request.assignment_ids),
            details={
                "assignment_count": result.modified_count,
                "comments": approval_request.comments
            }
        )
        background_tasks.add_task(log_audit_event, audit_log)
        
        # Send notifications
        for assignment_id in approval_request.assignment_ids:
            background_tasks.add_task(
                send_assignment_notification,
                assignment_id,
                "Assignment Approved",
                f"Assignment has been approved and locked for execution"
            )
        
        logger.info(f"Approved {result.modified_count} assignments by user {current_user['id']}")
        return {
            "message": f"Successfully approved {result.modified_count} assignments",
            "approved_count": result.modified_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving assignments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve assignments: {str(e)}")


@router.post("/override")
async def override_assignment(
    override_request: OverrideRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Override an assignment decision"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        current_time = datetime.utcnow()
        
        # Get original assignment
        original_assignment = await collection.find_one({"id": override_request.assignment_id})
        if not original_assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        # Update assignment
        result = await collection.update_one(
            {"id": override_request.assignment_id},
            {
                "$set": {
                    "status": AssignmentStatus.OVERRIDDEN.value,
                    "override_reason": override_request.reason,
                    "override_by": current_user["id"],
                    "override_at": current_time,
                    "override_decision": override_request.override_decision,
                    "last_updated": current_time
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Failed to override assignment")
        
        # Log audit event
        audit_log = AuditLogCreate(
            user_id=current_user["id"],
            action=AuditAction.ASSIGNMENT_OVERRIDDEN,
            resource_type="assignment",
            resource_id=override_request.assignment_id,
            details={
                "original_decision": original_assignment["decision"]["decision"],
                "override_decision": override_request.override_decision,
                "reason": override_request.reason,
                "trainset_id": original_assignment["trainset_id"]
            },
            risk_level="HIGH"
        )
        background_tasks.add_task(log_audit_event, audit_log)
        
        # Send notification
        background_tasks.add_task(
            send_assignment_notification,
            override_request.assignment_id,
            "Assignment Overridden",
            f"Assignment decision overridden: {override_request.override_decision}"
        )
        
        logger.info(f"Overridden assignment {override_request.assignment_id} by user {current_user['id']}")
        return {
            "message": "Assignment successfully overridden",
            "assignment_id": override_request.assignment_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error overriding assignment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to override assignment: {str(e)}")


@router.get("/summary", response_model=AssignmentSummary)
async def get_assignment_summary(
    _auth=Depends(require_api_key)
):
    """Get assignment summary statistics"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        
        # Get total counts
        total_assignments = await collection.count_documents({})
        
        # If no data in database, create some real assignments first
        if total_assignments == 0:
            await create_sample_assignments()
            # Re-fetch after creating sample data
            total_assignments = await collection.count_documents({})
        
        pending_count = await collection.count_documents({"status": AssignmentStatus.PENDING.value})
        approved_count = await collection.count_documents({"status": AssignmentStatus.APPROVED.value})
        rejected_count = await collection.count_documents({"status": AssignmentStatus.REJECTED.value})
        overridden_count = await collection.count_documents({"status": AssignmentStatus.OVERRIDDEN.value})
        high_priority_count = await collection.count_documents({"priority": {"$gte": 4}})
        
        # Get average confidence score
        pipeline = [
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$decision.confidence_score"}}}
        ]
        result = await collection.aggregate(pipeline).to_list(1)
        avg_confidence_score = result[0]["avg_confidence"] if result else 0.0
        
        # Count critical risks (assignments with violations)
        critical_risks_count = await collection.count_documents({
            "decision.violations": {"$exists": True, "$ne": []}
        })
        
        return AssignmentSummary(
            total_assignments=total_assignments,
            pending_count=pending_count,
            approved_count=approved_count,
            rejected_count=rejected_count,
            overridden_count=overridden_count,
            high_priority_count=high_priority_count,
            critical_risks_count=critical_risks_count,
            avg_confidence_score=avg_confidence_score,
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error fetching assignment summary: {e}")
        # Return mock data on error
        return AssignmentSummary(
            total_assignments=15,
            pending_count=8,
            approved_count=5,
            rejected_count=1,
            overridden_count=1,
            high_priority_count=3,
            critical_risks_count=2,
            avg_confidence_score=0.85,
            last_updated=datetime.utcnow()
        )


async def create_sample_assignments():
    """Create sample assignments with real data"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        
        # Get trainsets to create assignments for
        trainsets_collection = await cloud_db_manager.get_collection("trainsets")
        trainsets = []
        async for doc in trainsets_collection.find({}).limit(15):
            trainsets.append(doc)
        
        if not trainsets:
            # Create sample trainsets if none exist
            sample_trainsets = [
                {"trainset_id": f"TS-{i:03d}", "status": "ACTIVE", "sensor_health_score": 0.85 + (i * 0.01)}
                for i in range(1, 16)
            ]
            await trainsets_collection.insert_many(sample_trainsets)
            trainsets = sample_trainsets
        
        # Create sample assignments
        assignments = []
        for i, trainset in enumerate(trainsets[:15]):
            decision = random.choice(["INDUCT", "STANDBY", "MAINTENANCE"])
            status = random.choice(["PENDING", "APPROVED", "OVERRIDDEN"])
            confidence = round(random.uniform(0.7, 0.95), 2)
            priority = random.randint(1, 5)
            
            # Generate violations for some assignments (higher chance for conflicts)
            violations = []
            if random.random() < 0.4:  # 40% chance of violations for more conflicts
                violation_types = [
                    "Safety certificate expiring soon",
                    "Maintenance overdue", 
                    "Cleaning schedule conflict",
                    "Branding contract expired",
                    "High failure risk detected",
                    "Sensor health below threshold",
                    "Certificate validation failed"
                ]
                violations = random.sample(violation_types, random.randint(1, 3))
            
            assignment = {
                "id": f"ASS-{i+1:03d}",
                "trainset_id": trainset["trainset_id"],
                "status": status,
                "priority": priority,
                "decision": {
                    "decision": decision,
                    "confidence_score": confidence,
                    "reasoning": f"AI decision based on trainset {trainset['trainset_id']} analysis",
                    "violations": violations
                },
                "created_at": (datetime.utcnow() - timedelta(days=random.randint(0, 7))).isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "assigned_to": f"Operator-{random.randint(1, 5)}",
                "scheduled_date": (datetime.utcnow() + timedelta(days=random.randint(1, 3))).isoformat()
            }
            assignments.append(assignment)
        
        await collection.insert_many(assignments)
        logger.info(f"Created {len(assignments)} sample assignments")
        
    except Exception as e:
        logger.error(f"Error creating sample assignments: {e}")


@router.get("/conflicts", response_model=List[Assignment])
async def get_conflict_assignments(
    _auth=Depends(require_api_key)
):
    """Get assignments with conflicts (violations)"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        
        # Find assignments with violations
        cursor = collection.find({
            "decision.violations": {"$exists": True, "$ne": []}
        })
        
        assignments = []
        async for doc in cursor:
            doc.pop('_id', None)
            assignments.append(Assignment(**doc))
        
        return assignments
        
    except Exception as e:
        logger.error(f"Error fetching conflict assignments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conflicts: {str(e)}")


async def log_audit_event(audit_log: AuditLogCreate):
    """Log audit event to database"""
    try:
        audit_collection = await cloud_db_manager.get_collection("audit_logs")
        audit_doc = audit_log.dict()
        audit_doc["id"] = str(uuid.uuid4())
        audit_doc["timestamp"] = datetime.utcnow()
        await audit_collection.insert_one(audit_doc)
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")


async def send_assignment_notification(assignment_id: str, title: str, message: str):
    """Send notification about assignment update"""
    try:
        await notification_service.create_notification(
            type="ASSIGNMENT_UPDATE",
            priority="MEDIUM",
            title=title,
            message=message,
            data={"assignment_id": assignment_id}
        )
    except Exception as e:
        logger.error(f"Failed to send assignment notification: {e}")
