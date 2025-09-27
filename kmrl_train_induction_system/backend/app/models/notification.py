from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NotificationType(str, Enum):
    ALERT = "ALERT"
    ASSIGNMENT_UPDATE = "ASSIGNMENT_UPDATE"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    MAINTENANCE_REMINDER = "MAINTENANCE_REMINDER"
    COMPLIANCE_ALERT = "COMPLIANCE_ALERT"
    PERFORMANCE_UPDATE = "PERFORMANCE_UPDATE"


class NotificationPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class NotificationStatus(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


class NotificationChannel(str, Enum):
    IN_APP = "IN_APP"
    EMAIL = "EMAIL"
    SMS = "SMS"
    PUSH = "PUSH"
    WEBHOOK = "WEBHOOK"


class Notification(BaseModel):
    id: str = Field(..., description="Unique notification ID")
    type: NotificationType = Field(..., description="Notification type")
    priority: NotificationPriority = Field(..., description="Notification priority")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    
    # Recipients
    user_id: Optional[str] = Field(None, description="Target user ID")
    role: Optional[str] = Field(None, description="Target role")
    channels: List[NotificationChannel] = Field(default_factory=list, description="Delivery channels")
    
    # Content
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional notification data")
    action_url: Optional[str] = Field(None, description="URL for action button")
    action_text: Optional[str] = Field(None, description="Text for action button")
    
    # Status and delivery
    status: NotificationStatus = Field(default=NotificationStatus.PENDING, description="Notification status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    sent_at: Optional[datetime] = Field(None, description="Sent timestamp")
    delivered_at: Optional[datetime] = Field(None, description="Delivered timestamp")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledged timestamp")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    
    # Retry and failure handling
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    failure_reason: Optional[str] = Field(None, description="Failure reason if delivery failed")
    
    # Expiration
    expires_at: Optional[datetime] = Field(None, description="Notification expiration")
    ttl_seconds: int = Field(default=86400, description="Time to live in seconds (24 hours default)")


class NotificationCreate(BaseModel):
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    user_id: Optional[str] = None
    role: Optional[str] = None
    channels: List[NotificationChannel] = Field(default_factory=lambda: [NotificationChannel.IN_APP])
    data: Dict[str, Any] = Field(default_factory=dict)
    action_url: Optional[str] = None
    action_text: Optional[str] = None
    expires_at: Optional[datetime] = None
    ttl_seconds: int = 86400


class NotificationUpdate(BaseModel):
    status: Optional[NotificationStatus] = None
    acknowledged_by: Optional[str] = None
    failure_reason: Optional[str] = None


class NotificationFilter(BaseModel):
    user_id: Optional[str] = None
    type: Optional[NotificationType] = None
    priority: Optional[NotificationPriority] = None
    status: Optional[NotificationStatus] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    acknowledged: Optional[bool] = None


class NotificationSummary(BaseModel):
    total_notifications: int
    unread_count: int
    critical_count: int
    high_priority_count: int
    pending_count: int
    failed_count: int
    notifications_by_type: Dict[str, int]
    recent_notifications: List[Notification]


class Alert(BaseModel):
    id: str = Field(..., description="Unique alert ID")
    type: str = Field(..., description="Alert type (CRITICAL, HIGH, WARNING)")
    category: str = Field(..., description="Alert category")
    trainset_id: str = Field(..., description="Related trainset ID")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Alert timestamp")
    acknowledged: bool = Field(default=False, description="Whether alert is acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional alert data")
