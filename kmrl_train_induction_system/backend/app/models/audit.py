from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuditAction(str, Enum):
    # Assignment actions
    ASSIGNMENT_CREATED = "ASSIGNMENT_CREATED"
    ASSIGNMENT_APPROVED = "ASSIGNMENT_APPROVED"
    ASSIGNMENT_REJECTED = "ASSIGNMENT_REJECTED"
    ASSIGNMENT_OVERRIDDEN = "ASSIGNMENT_OVERRIDDEN"
    
    # Trainset actions
    TRAINSET_UPDATED = "TRAINSET_UPDATED"
    TRAINSET_STATUS_CHANGED = "TRAINSET_STATUS_CHANGED"
    
    # Optimization actions
    OPTIMIZATION_RUN = "OPTIMIZATION_RUN"
    OPTIMIZATION_OVERRIDE = "OPTIMIZATION_OVERRIDE"
    
    # User actions
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    USER_PERMISSION_CHANGED = "USER_PERMISSION_CHANGED"
    
    # System actions
    SYSTEM_CONFIGURATION_CHANGED = "SYSTEM_CONFIGURATION_CHANGED"
    DATA_INGESTION = "DATA_INGESTION"
    MODEL_TRAINING = "MODEL_TRAINING"
    
    # Report actions
    REPORT_GENERATED = "REPORT_GENERATED"
    REPORT_EXPORTED = "REPORT_EXPORTED"


class AuditLog(BaseModel):
    id: str = Field(..., description="Unique audit log ID")
    user_id: str = Field(..., description="User who performed the action")
    action: AuditAction = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: str = Field(..., description="ID of the resource affected")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Action timestamp")
    
    # Additional details
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional action details")
    ip_address: Optional[str] = Field(None, description="IP address of the user")
    user_agent: Optional[str] = Field(None, description="User agent string")
    session_id: Optional[str] = Field(None, description="User session ID")
    
    # Risk assessment
    risk_level: str = Field(default="LOW", description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    requires_review: bool = Field(default=False, description="Whether this action requires review")
    
    # Compliance
    compliance_category: Optional[str] = Field(None, description="Compliance category")
    retention_period_days: int = Field(default=2555, description="Retention period in days (7 years default)")


class AuditLogCreate(BaseModel):
    user_id: str
    action: AuditAction
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    risk_level: str = "LOW"
    requires_review: bool = False
    compliance_category: Optional[str] = None


class AuditLogFilter(BaseModel):
    user_id: Optional[str] = None
    action: Optional[AuditAction] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    risk_level: Optional[str] = None
    requires_review: Optional[bool] = None
    compliance_category: Optional[str] = None


class AuditSummary(BaseModel):
    total_actions: int
    actions_by_type: Dict[str, int]
    actions_by_user: Dict[str, int]
    high_risk_actions: int
    actions_requiring_review: int
    most_active_users: List[Dict[str, Any]]
    recent_actions: List[AuditLog]
    compliance_summary: Dict[str, int]
