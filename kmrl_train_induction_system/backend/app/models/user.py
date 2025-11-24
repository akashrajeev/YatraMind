from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, EmailStr


class UserRole(str, Enum):
    SUPERVISOR = "SUPERVISOR"
    MAINTENANCE_ENGINEER = "MAINTENANCE_ENGINEER"
    OPERATIONS_MANAGER = "OPERATIONS_MANAGER"
    READONLY_VIEWER = "READONLY_VIEWER"


class User(BaseModel):
    id: str = Field(..., description="Unique user ID")
    username: str = Field(..., description="Username for login")
    email: Optional[EmailStr] = Field(None, description="User email address (optional)")
    name: str = Field(..., description="User full name")
    role: UserRole = Field(..., description="User role")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_active: bool = Field(default=True, description="Whether user account is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    # Profile information
    department: Optional[str] = Field(None, description="User department")
    phone: Optional[str] = Field(None, description="User phone number")
    employee_id: Optional[str] = Field(None, description="Employee ID")


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    name: str
    role: UserRole
    permissions: List[str] = Field(default_factory=list)
    department: Optional[str] = None
    phone: Optional[str] = None
    employee_id: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    employee_id: Optional[str] = None


class UserLogin(BaseModel):
    username: str  # Changed from email to username
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, description="New password (minimum 8 characters)")


class Permission(BaseModel):
    name: str
    description: str
    category: str


# Predefined permissions
PERMISSIONS = {
    "assignments.view": Permission(
        name="assignments.view",
        description="View assignments",
        category="assignments"
    ),
    "assignments.approve": Permission(
        name="assignments.approve",
        description="Approve assignments",
        category="assignments"
    ),
    "assignments.override": Permission(
        name="assignments.override",
        description="Override assignment decisions",
        category="assignments"
    ),
    "trainsets.view": Permission(
        name="trainsets.view",
        description="View trainset information",
        category="trainsets"
    ),
    "trainsets.edit": Permission(
        name="trainsets.edit",
        description="Edit trainset information",
        category="trainsets"
    ),
    "reports.generate": Permission(
        name="reports.generate",
        description="Generate reports",
        category="reports"
    ),
    "reports.export": Permission(
        name="reports.export",
        description="Export data",
        category="reports"
    ),
    "system.admin": Permission(
        name="system.admin",
        description="System administration",
        category="system"
    ),
    "users.manage": Permission(
        name="users.manage",
        description="Manage users",
        category="users"
    ),
    "audit.view": Permission(
        name="audit.view",
        description="View audit logs",
        category="audit"
    ),
}

# Role-based permission mappings
ROLE_PERMISSIONS = {
    UserRole.OPERATIONS_MANAGER: list(PERMISSIONS.keys()),
    UserRole.SUPERVISOR: [
        "assignments.view", "assignments.approve", "assignments.override",
        "trainsets.view", "trainsets.edit", "reports.generate", "reports.export"
    ],
    UserRole.MAINTENANCE_ENGINEER: [
        "assignments.view", "trainsets.view", "trainsets.edit", "reports.generate"
    ],
    UserRole.READONLY_VIEWER: [
        "assignments.view", "trainsets.view", "reports.generate"
    ],
}
