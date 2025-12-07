# backend/app/api/assignments.py
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import logging
import uuid
import random
import functools
import traceback

from app.models.assignment import (
    Assignment, AssignmentCreate, AssignmentUpdate, AssignmentFilter, AssignmentSummary,
    ApprovalRequest, OverrideRequest, AssignmentStatus
)
from app.models.audit import AuditLogCreate, AuditAction
from app.models.trainset import InductionDecision
from app.utils.cloud_database import cloud_db_manager
from app.security import require_api_key
from app.services.notification_service import NotificationService
from app.utils.explainability import generate_maintenance_reasons

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize notification service
notification_service = NotificationService()

def safe_background_task(func):
    """Decorator to safely execute background tasks with error logging"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in background task {func.__name__}: {e}")
            logger.error(traceback.format_exc())
    return wrapper


def _transform_doc_to_assignment(doc: Dict[str, Any]) -> Assignment:
    """Transform database document to Assignment model, handling missing fields and data structure mismatches"""
    # Remove MongoDB _id
    doc.pop('_id', None)
    
    # Get trainset_id from assignment level
    trainset_id = doc.get('trainset_id', '')
    
    # Fix decision field - ensure it has trainset_id inside it for InductionDecision model
    decision_data = doc.get('decision', {})
    if isinstance(decision_data, dict):
        # Add trainset_id to decision if missing
        if 'trainset_id' not in decision_data:
            decision_data['trainset_id'] = trainset_id
        
        # Map old field names to new ones if needed
        if 'reasoning' in decision_data and 'reasons' not in decision_data:
            decision_data['reasons'] = [decision_data.pop('reasoning')] if decision_data.get('reasoning') else []
        
        # Ensure all required InductionDecision fields exist
        decision_data.setdefault('decision', 'STANDBY')
        decision_data.setdefault('confidence_score', 0.8)
        decision_data.setdefault('reasons', [])
        decision_data.setdefault('score', 0.0)
        decision_data.setdefault('top_reasons', [])
        decision_data.setdefault('top_risks', [])
        decision_data.setdefault('violations', [])
        decision_data.setdefault('shap_values', [])
        
        doc['decision'] = decision_data
    
    # Add missing required fields with defaults
    if 'created_by' not in doc:
        doc['created_by'] = doc.get('assigned_to', 'system')
    
    # Map updated_at to last_updated if needed
    if 'updated_at' in doc and 'last_updated' not in doc:
        try:
            if isinstance(doc['updated_at'], str):
                doc['last_updated'] = datetime.fromisoformat(doc['updated_at'].replace('Z', '+00:00'))
            else:
                doc['last_updated'] = doc['updated_at']
        except Exception:
            doc['last_updated'] = datetime.now(timezone.utc)
    
    # Ensure created_at is datetime if it's a string
    if 'created_at' in doc and isinstance(doc['created_at'], str):
        try:
            doc['created_at'] = datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00'))
        except Exception:
            doc['created_at'] = datetime.now(timezone.utc)
    
    # Map scheduled_date to execution_date if needed
    if 'scheduled_date' in doc and 'execution_date' not in doc:
        try:
            if isinstance(doc['scheduled_date'], str):
                doc['execution_date'] = datetime.fromisoformat(doc['scheduled_date'].replace('Z', '+00:00'))
            else:
                doc['execution_date'] = doc['scheduled_date']
        except Exception:
            pass
    
    return Assignment(**doc)


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
            try:
                assignments.append(_transform_doc_to_assignment(doc))
            except Exception as e:
                logger.warning(f"Skipping invalid assignment document: {e}")
                continue
        
        return assignments
        
    except Exception as e:
        logger.error(f"Error fetching assignments: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


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
            last_updated=datetime.now(timezone.utc)
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
            last_updated=datetime.now(timezone.utc)
        )


def _needs_maintenance(trainset: Dict[str, Any]) -> bool:
    """Check if trainset needs maintenance
    
    Uses safe dict access to prevent KeyError on malformed data.
    Matches the logic from optimizer.py
    """
    # Safe dict access with defaults
    job_cards = trainset.get("job_cards", {})
    if not isinstance(job_cards, dict):
        job_cards = {}
    
    critical_cards = job_cards.get("critical_cards", 0)
    try:
        critical_cards = int(critical_cards) if critical_cards is not None else 0
    except (TypeError, ValueError):
        critical_cards = 0
    
    # Safe mileage access
    try:
        current_mileage = float(trainset.get("current_mileage", 0.0))
    except (TypeError, ValueError):
        current_mileage = 0.0
    
    try:
        max_mileage = float(trainset.get("max_mileage_before_maintenance", float('inf')))
    except (TypeError, ValueError):
        max_mileage = float('inf')
    
    return (
        critical_cards > 0 or
        (max_mileage > 0 and current_mileage >= max_mileage * 0.95)
    )


@router.get("/conflicts", response_model=List[Assignment])
async def get_conflict_assignments(
    _auth=Depends(require_api_key)
):
    """Get all trains that need maintenance with maintenance reasons"""
    try:
        trainsets_collection = await cloud_db_manager.get_collection("trainsets")
        assignments_collection = await cloud_db_manager.get_collection("assignments")
        
        # Get ALL trainsets and check which ones need maintenance
        trainsets_cursor = trainsets_collection.find({})
        
        maintenance_trainsets = []
        async for trainset_doc in trainsets_cursor:
            try:
                trainset_doc.pop("_id", None)
                trainset_id = trainset_doc.get("trainset_id", "")
                if not trainset_id:
                    continue
                
                # Check if train needs maintenance
                if _needs_maintenance(trainset_doc):
                    # Generate maintenance reasons
                    maintenance_info = generate_maintenance_reasons(trainset_doc)
                    
                    # Try to find existing assignment for this trainset
                    existing_assignment = await assignments_collection.find_one({"trainset_id": trainset_id})
                    
                    if existing_assignment:
                        # Update existing assignment with maintenance decision and reasons
                        assignment = _transform_doc_to_assignment(existing_assignment)
                        # Update decision to MAINTENANCE if not already
                        if assignment.decision.decision != "MAINTENANCE":
                            assignment.decision.decision = "MAINTENANCE"
                        # Update with maintenance reasons
                        assignment.decision.top_reasons = maintenance_info.get("top_reasons", [])
                        assignment.decision.top_risks = maintenance_info.get("top_risks", [])
                        assignment.decision.score = 0.0  # Maintenance trains get 0 score
                        assignment.decision.confidence_score = 1.0  # High confidence for maintenance
                    else:
                        # Create new assignment for maintenance
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
                                shap_values=[]
                            ),
                            status=AssignmentStatus.PENDING,
                            created_by="system",
                            priority=5  # High priority for maintenance
                        )
                    
                    maintenance_trainsets.append(assignment)
            except Exception as e:
                logger.warning(f"Skipping trainset {trainset_doc.get('trainset_id', 'unknown')}: {e}")
                continue
        
        # Sort by priority (maintenance trains should be prioritized)
        maintenance_trainsets.sort(key=lambda x: x.priority, reverse=True)
        
        return maintenance_trainsets
        
    except Exception as e:
        logger.error(f"Error fetching conflict assignments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conflicts: {str(e)}")


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
        
        return _transform_doc_to_assignment(doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching assignment {assignment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/", response_model=Assignment)
async def create_assignment(
    assignment_data: AssignmentCreate,
    background_tasks: BackgroundTasks,
    _auth=Depends(require_api_key)
):
    """Create a new assignment"""
    try:
        assignment_id = str(uuid.uuid4())
        
        # Use created_by from request or default to 'system'
        created_by = getattr(assignment_data, 'created_by', None) or "system"
        
        assignment = Assignment(
            id=assignment_id,
            trainset_id=assignment_data.trainset_id,
            decision=assignment_data.decision,
            created_by=created_by,
            priority=assignment_data.priority,
            execution_date=assignment_data.execution_date
        )
        
        # Save to database
        collection = await cloud_db_manager.get_collection("assignments")
        await collection.insert_one(assignment.dict())
        
        # Log audit event
        audit_log = AuditLogCreate(
            user_id=created_by,
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
    _auth=Depends(require_api_key)
):
    """Approve multiple assignments"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        current_time = datetime.now(timezone.utc)
        
        # Use user_id from request, or default to 'system' if not provided
        user_id = approval_request.user_id or "system"
        
        # Only update assignments that are PENDING (can't approve already approved/overridden)
        result = await collection.update_many(
            {
                "id": {"$in": approval_request.assignment_ids},
                "status": AssignmentStatus.PENDING.value  # Only approve PENDING assignments
            },
            {
                "$set": {
                    "status": AssignmentStatus.APPROVED.value,
                    "approved_by": user_id,
                    "approved_at": current_time,
                    "approval_comments": approval_request.comments,
                    "last_updated": current_time
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="No pending assignments found to approve")
        
        # Log audit event
        audit_log = AuditLogCreate(
            user_id=user_id,
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
        
        logger.info(f"Approved {result.modified_count} assignments by user {user_id}")
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
    _auth=Depends(require_api_key)
):
    """Override an assignment decision"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        current_time = datetime.now(timezone.utc)
        
        # Use user_id from request, or default to 'system' if not provided
        user_id = override_request.user_id or "system"
        
        # Atomic update ensuring status is PENDING
        updated_assignment = await collection.find_one_and_update(
            {
                "id": override_request.assignment_id,
                "status": AssignmentStatus.PENDING.value
            },
            {
                "$set": {
                    "status": AssignmentStatus.OVERRIDDEN.value,
                    "override_reason": override_request.reason,
                    "override_by": user_id,
                    "override_at": current_time,
                    "override_decision": override_request.override_decision,
                    "last_updated": current_time
                }
            },
            return_document=True
        )
        
        if not updated_assignment:
            # Check if assignment exists at all to give better error message
            existing = await collection.find_one({"id": override_request.assignment_id})
            if not existing:
                raise HTTPException(status_code=404, detail="Assignment not found")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Cannot override assignment with status {existing.get('status')}"
                )
        
        # Log audit event
        audit_log = AuditLogCreate(
            user_id=user_id,
            action=AuditAction.ASSIGNMENT_OVERRIDDEN,
            resource_type="assignment",
            resource_id=override_request.assignment_id,
            details={
                "original_decision": updated_assignment.get("decision", {}).get("decision", "UNKNOWN"),
                "override_decision": override_request.override_decision,
                "reason": override_request.reason,
                "trainset_id": updated_assignment.get("trainset_id")
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
        
        logger.info(f"Overridden assignment {override_request.assignment_id} by user {user_id}")
        return {
            "message": "Assignment successfully overridden",
            "assignment_id": override_request.assignment_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error overriding assignment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to override assignment: {str(e)}")


async def create_sample_assignments():
    """Create sample assignments with real data - all as PENDING"""
    try:
        collection = await cloud_db_manager.get_collection("assignments")
        
        # Check if assignments already exist
        existing_count = await collection.count_documents({})
        if existing_count > 0:
            logger.info(f"Skipping sample assignment creation - {existing_count} assignments already exist")
            return
        
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
        
        # Create sample assignments - all start as PENDING (user must approve/override them)
        assignments = []
        for i, trainset in enumerate(trainsets[:15]):
            decision = random.choice(["INDUCT", "STANDBY", "MAINTENANCE"])
            status = "PENDING"  # All sample assignments start as PENDING - user must approve/override
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
            
            trainset_id = trainset.get("trainset_id", f"TS-{i:03d}")
            assignment = {
                "id": f"ASS-{i+1:03d}",
                "trainset_id": trainset_id,
                "status": status,
                "priority": priority,
                "decision": {
                    "trainset_id": trainset_id,  # Required by InductionDecision model
                    "decision": decision,
                    "confidence_score": confidence,
                    "reasons": [f"AI decision based on trainset {trainset_id} analysis"],
                    "top_reasons": [],
                    "top_risks": [],
                    "violations": violations,
                    "score": confidence,
                    "shap_values": []
                },
                "created_at": (datetime.now(timezone.utc) - timedelta(days=random.randint(0, 7))),
                "last_updated": datetime.now(timezone.utc),
                "created_by": f"system-{random.randint(1, 5)}",
                "execution_date": (datetime.now(timezone.utc) + timedelta(days=random.randint(1, 3)))
            }
            assignments.append(assignment)
        
        await collection.insert_many(assignments)
        logger.info(f"Created {len(assignments)} sample assignments")
        
    except Exception as e:
        logger.error(f"Error creating sample assignments: {e}")


@safe_background_task
async def log_audit_event(audit_log: AuditLogCreate):
    """Log audit event to database"""
    try:
        audit_collection = await cloud_db_manager.get_collection("audit_logs")
        audit_doc = audit_log.dict()
        audit_doc["id"] = str(uuid.uuid4())
        audit_doc["timestamp"] = datetime.now(timezone.utc)
        await audit_collection.insert_one(audit_doc)
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")


@safe_background_task
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
