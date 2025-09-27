from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from app.models.trainset import InductionDecision


class AssignmentStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    OVERRIDDEN = "OVERRIDDEN"


class Assignment(BaseModel):
    id: str = Field(..., description="Unique assignment ID")
    trainset_id: str = Field(..., description="Trainset identifier")
    decision: InductionDecision = Field(..., description="AI/ML decision details")
    status: AssignmentStatus = Field(default=AssignmentStatus.PENDING, description="Assignment status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    created_by: str = Field(..., description="User who created the assignment")
    
    # Approval fields
    approved_by: Optional[str] = Field(None, description="User who approved the assignment")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    approval_comments: Optional[str] = Field(None, description="Approval comments")
    
    # Override fields
    override_reason: Optional[str] = Field(None, description="Reason for override")
    override_by: Optional[str] = Field(None, description="User who overrode the assignment")
    override_at: Optional[datetime] = Field(None, description="Override timestamp")
    override_decision: Optional[str] = Field(None, description="Override decision (INDUCT/STANDBY/MAINTENANCE)")
    
    # Metadata
    priority: int = Field(default=1, description="Assignment priority (1-5)")
    execution_date: Optional[datetime] = Field(None, description="Scheduled execution date")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class AssignmentCreate(BaseModel):
    trainset_id: str
    decision: InductionDecision
    created_by: str
    priority: int = 1
    execution_date: Optional[datetime] = None


class AssignmentUpdate(BaseModel):
    status: Optional[AssignmentStatus] = None
    approved_by: Optional[str] = None
    approval_comments: Optional[str] = None
    override_reason: Optional[str] = None
    override_by: Optional[str] = None
    override_decision: Optional[str] = None
    priority: Optional[int] = None
    execution_date: Optional[datetime] = None


class ApprovalRequest(BaseModel):
    assignment_ids: List[str] = Field(..., description="List of assignment IDs to approve")
    user_id: str = Field(..., description="User ID of approver")
    comments: Optional[str] = Field(None, description="Approval comments")


class OverrideRequest(BaseModel):
    assignment_id: str = Field(..., description="Assignment ID to override")
    user_id: str = Field(..., description="User ID of overrider")
    reason: str = Field(..., description="Override reason")
    override_decision: str = Field(..., description="New decision (INDUCT/STANDBY/MAINTENANCE)")


class AssignmentFilter(BaseModel):
    status: Optional[AssignmentStatus] = None
    trainset_id: Optional[str] = None
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    priority: Optional[int] = None
    execution_date_after: Optional[datetime] = None
    execution_date_before: Optional[datetime] = None


class AssignmentSummary(BaseModel):
    total_assignments: int
    pending_count: int
    approved_count: int
    rejected_count: int
    overridden_count: int
    high_priority_count: int
    critical_risks_count: int
    avg_confidence_score: float
    last_updated: datetime
